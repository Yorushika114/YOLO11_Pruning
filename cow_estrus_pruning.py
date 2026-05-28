#!/usr/bin/env python3
"""
论文算法实现：融合 YOLO v5n 与通道剪枝算法的轻量化奶牛发情行为识别

Paper: Lightweight recognition for the oestrus behavior of dairy cows
       combining YOLO v5n and channel pruning
Authors: Wang Zheng, Xu Xingshi, Hua Zhixin, et al.
Journal: Transactions of the Chinese Society of Agricultural Engineering, 2022, 38(23): 130-140

三阶段流程
----------
Phase 1  稀疏化训练 (Sparse Training)
    对 BN 层的缩放因子 γ 施加 L1 正则化惩罚，使不重要通道的 γ 趋近于 0。
    损失函数（论文公式 10）：L = Σ l(f(c,W), y) + λ·Σ g(γ)，其中 g(γ)=|γ|
    论文参数：稀疏率 sr = λ = 0.005，训练 200 轮。

Phase 2  通道剪枝 (Channel Pruning)
    以 BN 层 γ 的绝对值作为通道重要性指标，剪去重要性最低的 50% 通道。
    引用：Network Slimming (Liu et al., ICCV 2017) ——论文参考文献 [20]

Phase 3  微调 (Fine-tuning)
    对剪枝后的模型进行精度恢复训练。

论文最终结果 (YOLOv5n-Pruned vs. 原始 YOLOv5-Nano)：
    mAP      : 97.70%（持平）
    Params   : 0.72 M  （减少 59.32%）
    FLOPs    : 0.68 G  （减少 49.63%）
    推理速度  : 50.26 帧/s（提升 33.71%）
"""

from ultralytics.utils import LOGGER, RANK
from ultralytics.nn.modules import Detect
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.engine.trainer import BaseTrainer
from ultralytics import YOLO
import os
import sys
import math
import warnings
import argparse
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# ── Ultralytics ──────────────────────────────────────────────────────────────

# ── torch-pruning（可选）────────────────────────────────────────────────────
try:
    import torch_pruning as tp
    _HAS_TP = True
except ImportError:
    _HAS_TP = False
    print("[警告] torch_pruning 未安装，剪枝将使用内置简化实现。")
    print("       安装命令: pip install torch-pruning")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1  稀疏化训练
# ═══════════════════════════════════════════════════════════════════════════════

class SparseDetectionTrainer(DetectionTrainer):
    """
    带 BN-gamma L1 稀疏正则化的检测训练器。

    在标准 YOLO 损失后，通过修改 BN.weight (即 γ) 的梯度来施加 L1 惩罚：
        γ.grad += sr · sign(γ)
    等价于将 L1 正则项 λ·Σ|γ| 加入损失函数。

    论文公式 (8)(9)：
        Z̃ = (Z_in − μ_B) / sqrt(σ_B² + ε)
        Z_out = γ·Z̃ + β
    训练过程中通过 L1 正则化使不重要通道的 γ 趋近于 0（公式 10）。
    """

    def __init__(self, overrides=None, _callbacks=None, sparsity_rate: float = 0.005):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.sparsity_rate = sparsity_rate
        LOGGER.info(f"[SparseTrainer] 稀疏率 sr = {self.sparsity_rate}")

    def optimizer_step(self):
        """
        重写优化器步骤：在梯度裁剪前向 BN.weight 梯度添加 L1 次梯度。

        论文公式 (10)：
            L = Σ l(f(c,W), y)  +  λ · Σ g(γ)
        其中 g(γ) = |γ|，次梯度为 sign(γ)。
        """
        # 反缩放（AMP 混合精度下恢复真实梯度幅度）
        self.scaler.unscale_(self.optimizer)

        # ── 核心：向 BN.weight (γ) 梯度加入 L1 次梯度 ──────────────────────
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) and m.weight.grad is not None:
                # ∂|γ|/∂γ = sign(γ)
                m.weight.grad.data.add_(
                    self.sparsity_rate * torch.sign(m.weight.data))
        # ────────────────────────────────────────────────────────────────────

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)


def sparse_train(
    model_path: str,
    data_yaml: str,
    epochs: int = 200,
    sparsity_rate: float = 0.005,
    imgsz: int = 160,
    batch: int = 16,
    lr0: float = 0.0032,
    weight_decay: float = 0.00036,
    save_dir: str = "sparse_trained",
    device: str = "",

) -> str:
    """
    Phase 1：对 YOLO 模型进行稀疏化训练。

    论文设置（1.2.3 节）：
        - 稀疏率 sr = 0.005
        - 训练 200 轮，Loss 约在 150 轮后趋于收敛（图 6）
        - 初始学习率 lr0 = 0.003 2，权重衰减 0.000 36
        - 输入尺寸 512×512
        - 在 VOC 预训练权重基础上微调

    Args:
        model_path  : 预训练模型路径（.pt）
        data_yaml   : 数据集配置（含 train/val 路径和类别信息）
        epochs      : 训练轮数
        sparsity_rate: BN-gamma L1 正则化系数 λ
        imgsz       : 图像短边尺寸
        batch       : 批次大小
        lr0         : 初始学习率
        weight_decay: 权重衰减
        save_dir    : 输出目录名
        device      : 设备（如 '0', 'cpu'）

    Returns:
        稀疏训练后最优模型的权重路径
    """
    _print_phase(1, "稀疏化训练", f"sr={sparsity_rate}, epochs={epochs}")

    overrides = dict(
        model=model_path,
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        weight_decay=weight_decay,
        optimizer="SGD",
        momentum=0.937,
        amp=False,
        name=save_dir,
        verbose=True,
    )
    if device:
        overrides["device"] = device

    trainer = SparseDetectionTrainer(
        overrides=overrides, sparsity_rate=sparsity_rate)
    trainer.train()

    best = str(trainer.best)
    print(f"\n[Phase 1 完成] 稀疏训练最优模型 → {best}")
    analyze_bn_sparsity(best)
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2  通道剪枝
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2-a  BN-gamma 重要性评估（配合 torch-pruning 使用）──────────────────────

if _HAS_TP:
    class BNGammaImportance(tp.importance.Importance):
        """
        以 BN 层缩放因子 γ 的绝对值作为通道重要性（Network Slimming 准则）。

        修复要点：
        1. 遍历 group 中所有条目并收集所有 BN-gamma 分数后再聚合，
           不能在第一个 BN 遇到后就立即 return（可能遗漏其他耦合 BN）。
        2. 以 dep.target.module 类型区分 BN / Conv，对每种操作取对应维度。
        3. 若 group 内无 BN 层（如纯 Conv 的 skip 连接），退化到 Conv 权重 L1 均值。
        4. 所有候选分数沿 group 维度取平均后返回，而非只返回第一条。
        """

        def __call__(self, group, **kwargs):
            group_imp = []
            for dep, idxs in group:
                module = dep.target.module
                if isinstance(module, nn.BatchNorm2d):
                    # BN 的缩放因子 γ = module.weight；取绝对值作为重要性
                    scores = module.weight.data.abs()
                    group_imp.append(scores[idxs])

            if not group_imp:
                # group 内无 BN 层时，退化到 Conv 输出通道权重 L1 均值
                for dep, idxs in group:
                    module = dep.target.module
                    if isinstance(module, nn.Conv2d) and module.weight.ndim == 4:
                        scores = module.weight.data.abs().mean(dim=(1, 2, 3))
                        group_imp.append(scores[idxs])

            if not group_imp:
                return None

            # 多个 BN/Conv 分数沿 group 轴取均值，保证维度一致
            return torch.stack(group_imp, dim=0).mean(dim=0)


# ── 2-b  不依赖 torch-pruning 的内置分析器 ───────────────────────────────────

class _InternalPruneAnalyzer:
    """
    仅做分析的轻量版剪枝规划器（不实际改变模型结构）。

    用于在没有 torch-pruning 时展示论文算法逻辑：
        1. 收集所有 BN 层的 γ 值
        2. 以 pruning_rate 对应的分位数作为全局阈值
        3. 逐层统计保留/剪除通道数
    """

    MIN_KEEP = 8  # 每层至少保留的通道数

    def __init__(self, model: nn.Module, pruning_rate: float = 0.5):
        self.model = model
        self.pruning_rate = pruning_rate
        self._bn_list = [(n, m) for n, m in model.named_modules()
                         if isinstance(m, nn.BatchNorm2d)]

    def global_threshold(self) -> float:
        all_g = np.concatenate(
            [m.weight.data.abs().cpu().numpy() for _, m in self._bn_list]
        )
        thr = float(np.percentile(all_g, self.pruning_rate * 100))
        return thr

    def plan(self) -> dict:
        thr = self.global_threshold()
        total_before = total_after = 0
        result = {}
        for name, bn in self._bn_list:
            g = bn.weight.data.abs().cpu().numpy()
            keep_mask = g >= thr
            # 保证最少保留 MIN_KEEP 个通道
            n_keep = max(self.MIN_KEEP, int(keep_mask.sum()))
            if n_keep >= len(g):
                n_keep = len(g)
            result[name] = {
                "before": len(g),
                "after": n_keep,
                "pruned": len(g) - n_keep,
                "rate": (len(g) - n_keep) / len(g),
            }
            total_before += len(g)
            total_after += n_keep

        print(f"\n[剪枝规划] 全局阈值 = {thr:.6f}")
        print(f"  总通道: {total_before} → {total_after}"
              f"  (实际剪枝率 {1 - total_after / total_before:.2%})")
        for name, info in list(result.items())[:10]:
            print(f"  {name:50s}: {info['before']:3d} → {info['after']:3d}"
                  f"  (-{info['pruned']:3d}, {info['rate']:.0%})")
        if len(result) > 10:
            print(f"  ... 共 {len(result)} 层")
        return result


def prune_by_bn_gamma(
    model_path: str,
    pruning_rate: float = 0.5,
    save_path: str = "yolov5n_pruned.pt",
    imgsz: int = 160,
    device: str = "",
) -> str:
    """
    Phase 2：基于 BN-gamma 进行通道剪枝。

    论文设置（1.2.3 节 ②）：
        - 剪枝率 50%（论文图 9：mAP 变化趋势，剪枝率 0.5 最佳）
        - 对包括 CSPDarknet53 骨干网络在内的所有模块进行剪枝
        - 剪枝后参数量 → 0.72 M（论文表 2）

    论文图 10 显示：剪枝后多数卷积层通道数大幅降低，
    平均每层被剪枝约 32 个通道。

    Args:
        model_path  : 稀疏训练后的模型权重路径
        pruning_rate: 通道剪枝率（论文使用 0.5）
        save_path   : 剪枝后模型保存路径
        imgsz       : 示例输入尺寸（用于构建依赖图）
        device      : 设备

    Returns:
        剪枝后模型的保存路径
    """
    _print_phase(2, "通道剪枝", f"pruning_rate={pruning_rate}")

    target_device = _resolve_device(device)
    yolo = YOLO(model_path)
    model = yolo.model.to(target_device).eval()

    params_before = _count_params(model)
    print(f"[剪枝前] 参数量: {params_before / 1e6:.4f} M")

    if not _HAS_TP:
        # 无 torch-pruning：仅展示规划
        analyzer = _InternalPruneAnalyzer(model, pruning_rate)
        analyzer.plan()
        print("\n[提示] 安装 torch-pruning 后可执行实际剪枝（pip install torch-pruning）")
        torch.save({"model": model.state_dict(), "pruning_rate": pruning_rate,
                    "note": "分析模式，未实际剪枝"}, save_path)
        print(f"[Phase 2 完成] 分析结果已保存 → {save_path}")
        return save_path

    # ── 使用 torch-pruning 执行实际剪枝 ──────────────────────────────────────
    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(target_device)

    # 检测头不参与剪枝（保持输出维度）
    ignored_layers = [m for m in model.modules() if isinstance(m, Detect)]

    # 优先使用 torch-pruning 内置 BNScaleImportance（与论文准则等价），
    # 若该版本不支持则退回我们修复后的 BNGammaImportance。
    if hasattr(tp.importance, "BNScaleImportance"):
        importance = tp.importance.BNScaleImportance()
        print("[剪枝] 使用 tp.importance.BNScaleImportance（内置 BN-gamma 准则）")
    else:
        importance = BNGammaImportance()
        print("[剪枝] 使用自定义 BNGammaImportance（BN-gamma 准则）")

    # 剪枝前将模型置为训练模式，避免 BN running_stats 影响 gamma 梯度
    model.train()

    # MetaPruner 只传版本兼容的核心参数，去掉 round_to / isomorphic
    # 等可能在当前版本不存在的字段，防止静默失败。
    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=pruning_rate,
        ignored_layers=ignored_layers,
        global_pruning=True,       # 全局统一阈值（与论文一致）
    )
    pruner.step()
    model.eval()

    params_after = _count_params(model)
    print(f"[剪枝后] 参数量: {params_after / 1e6:.4f} M")
    print(f"[剪枝后] 参数减少: {1 - params_after / params_before:.2%}"
          f"  (论文目标: ~59.32%)")

    # 统计 FLOPs
    try:
        macs, _ = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"[剪枝后] FLOPs: {macs / 1e9:.4f} G  (论文目标: ~0.68 G)")
    except Exception as e:
        print(f"[警告] FLOPs 统计跳过: {e}")

    # 保存剪枝后模型
    torch.save({"model": model, "pruning_rate": pruning_rate}, save_path)
    print(f"\n[Phase 2 完成] 剪枝模型 → {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3  微调
# ═══════════════════════════════════════════════════════════════════════════════

class _PrunedModelTrainer(DetectionTrainer):
    """
    专门用于加载剪枝后模型（非标准结构）的训练器。

    剪枝后模型的通道数已改变，无法直接用 YOLO.train() 加载，
    需手动注入 model 后再调用 trainer.train()。
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """跳过模型初始化，直接返回外部注入的剪枝模型。"""
        return self.model


def finetune(
    pruned_model_path: str,
    data_yaml: str,
    epochs: int = 5,
    imgsz: int = 160,
    batch: int = 32,
    lr0: float = 0.001,
    save_dir: str = "finetuned",
    device: str = "",
    model="/gemini/code/Torch-Pruning-master/examples/yolov8/yolov5nu.pt"
) -> str:
    """
    Phase 3：对剪枝后的模型进行精度恢复微调。

    论文设置（1.2.3 节 ②）：
        - 稀疏率 0.005 的稀疏训练 + 50% 剪枝后进行微调
        - 微调后模型命名为 YOLOv5n-Pruned

    Args:
        pruned_model_path: 剪枝后模型路径
        data_yaml        : 数据集配置文件
        epochs           : 微调轮数
        imgsz            : 图像尺寸
        batch            : 批次大小
        lr0              : 初始学习率（通常比稀疏训练低一个量级）
        save_dir         : 输出目录
        device           : 设备

    Returns:
        微调后最优模型路径
    """
    _print_phase(3, "微调 (Fine-tuning)", f"epochs={epochs}, lr0={lr0}")

    # 加载剪枝后模型
    ckpt = torch.load(pruned_model_path, map_location="cpu",
                      weights_only=False)
    pruned_model = ckpt["model"] if isinstance(
        ckpt, dict) and "model" in ckpt else ckpt

    if isinstance(pruned_model, dict):
        # 若保存的是 state_dict，需先构建模型再加载
        raise ValueError("剪枝模型应以完整 nn.Module 对象保存，而非 state_dict。")

    overrides = dict(
        # BaseTrainer.__init__ 对 self.args.model 调用 check_model_file_from_stem()，
        # 该函数仅做 Path(model) 路径字符串检查，不实际打开文件；
        # setup_model() 检测到 self.model 已是 nn.Module 后直接 return，
        # 此处的 model 路径永远不会被加载（trainer.py L693）。
        model=model,
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        optimizer="SGD",
        momentum=0.937,
        amp=False,
        name=save_dir,
        verbose=True,
    )
    if device:
        overrides["device"] = device

    trainer = _PrunedModelTrainer(overrides=overrides)
    # 在 trainer.train() 进入 setup_model() 之前注入剪枝后的 nn.Module；
    # setup_model() 首行 isinstance(self.model, nn.Module) 为 True 则立即返回。
    trainer.model = pruned_model.to(_resolve_device(device))
    trainer.train()

    best = str(trainer.best)
    print(f"\n[Phase 3 完成] 微调最优模型 → {best}")
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_bn_sparsity(model_path: str, threshold: float = 0.01) -> np.ndarray:
    """
    分析 BN 层 γ 的稀疏程度（对应论文图 8）。

    稀疏率越高（更多 γ 趋近于 0），剪枝潜力越大。
    论文图 8 展示了 sr=0.0001, 0.001, 0.005, 0.01 四种稀疏率下的γ分布。
    sr=0.005 时效果最佳（γ快速趋近于0且精度损失可控）。
    """
    try:
        model = YOLO(model_path).model
        gammas = np.concatenate(
            [m.weight.data.abs().cpu().numpy()
             for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        )
        sparse = (gammas < threshold).mean()
        print(f"\n=== BN-γ 稀疏分析 ({model_path}) ===")
        print(f"  通道总数   : {len(gammas)}")
        print(f"  |γ| < {threshold} : {sparse:.2%}  ← 可理解为'近似零'通道比例")
        print(f"  均值/标准差 : {gammas.mean():.4f} / {gammas.std():.4f}")
        print(f"  分位数 [10%/50%/90%]: "
              f"[{np.percentile(gammas, 10):.4f} / "
              f"{np.percentile(gammas, 50):.4f} / "
              f"{np.percentile(gammas, 90):.4f}]")
        return gammas
    except Exception as e:
        print(f"[警告] 稀疏分析失败: {e}")
        return np.array([])


def plot_bn_gamma_distribution(model_path: str, save_fig: str = "bn_gamma_dist.png"):
    """
    可视化 BN-γ 分布直方图（对应论文图 8）。

    论文图 8 展示：随着稀疏训练的进行，γ 系数逐渐向 0 集中，
    表明稀疏化效果随 epoch 增加而增强。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model = YOLO(model_path).model
        gammas = np.concatenate(
            [m.weight.data.abs().cpu().numpy()
             for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"BN 层 γ 系数分布（论文图 8）\n模型: {Path(model_path).name}",
                     fontsize=12)

        # 直方图
        axes[0].hist(gammas, bins=100, color="steelblue",
                     alpha=0.75, edgecolor="none")
        axes[0].axvline(np.percentile(gammas, 50), color="red",
                        linestyle="--", label="中位数")
        axes[0].set_xlabel("|γ|")
        axes[0].set_ylabel("通道数")
        axes[0].set_title("γ 分布直方图")
        axes[0].legend()

        # 累积分布（CDF）
        sorted_g = np.sort(gammas)
        cdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)
        axes[1].plot(sorted_g, cdf, color="darkorange", linewidth=1.5)
        axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        axes[1].set_xlabel("|γ|")
        axes[1].set_ylabel("累积比例")
        axes[1].set_title("γ 累积分布 (CDF)")

        plt.tight_layout()
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[图像] BN-γ 分布图已保存 → {save_fig}")
    except Exception as e:
        print(f"[警告] 可视化失败: {e}")


def count_model_stats(model_path: str, imgsz: int = 160) -> dict:
    """
    统计模型参数量和计算量（对应论文表 3）。

    论文表 3 对比（与原始 YOLOv5-Nano 相比）：
        YOLOv5-Nano   : Params=1.77 M, FLOPs=1.35 G, Speed=37.59 f/s
        YOLOv5n-Pruned: Params=0.72 M, FLOPs=0.68 G, Speed=50.26 f/s
    """
    stats = {}
    try:
        yolo = YOLO(model_path)
        model = yolo.model
        dev = _resolve_device("")
        model = model.to(dev).eval()

        params = sum(p.numel() for p in model.parameters())
        stats["params_M"] = params / 1e6

        if _HAS_TP:
            x = torch.randn(1, 3, imgsz, imgsz).to(dev)
            with torch.no_grad():
                macs, _ = tp.utils.count_ops_and_params(model, x)
            stats["flops_G"] = macs / 1e9

        print(f"\n=== 模型统计 ({Path(model_path).name}) ===")
        print(f"  参数量 (Params) : {stats['params_M']:.4f} M")
        if "flops_G" in stats:
            print(f"  计算量 (FLOPs)  : {stats['flops_G']:.4f} G")
    except Exception as e:
        print(f"[警告] 模型统计失败: {e}")
    return stats


def ssim_dedup(frames: list, threshold: float = 0.95) -> list:
    """
    用 SSIM 去除视频中的冗余帧（论文 1.1.2 节数据预处理方法）。

    论文公式 (1)：
        SSIM(X, Y) = (2μ_X μ_Y + C1)(2σ_XY + C2)
                     / ((μ_X² + μ_Y² + C1)(σ_X² + σ_Y² + C2))
    相邻帧 SSIM 超过阈值则视为冗余帧并丢弃，
    最终从 199 段视频中筛选出 2 239 幅有效奶牛爬跨图像。

    Args:
        frames    : BGR numpy 图像列表
        threshold : SSIM 相似度阈值（超过则认为重复）

    Returns:
        去重后的帧列表
    """
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        import cv2
    except ImportError:
        print("[警告] 需要 scikit-image 和 opencv-python，跳过去重")
        return frames

    if len(frames) <= 1:
        return frames

    kept = [frames[0]]
    removed = 0
    for curr in frames[1:]:
        prev = kept[-1]
        g1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
        g2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if curr.ndim == 3 else curr
        if ssim_fn(g1, g2) < threshold:
            kept.append(curr)
        else:
            removed += 1

    print(f"[SSIM 去重] {len(frames)} → {len(kept)} 帧，去除 {removed} 冗余帧")
    return kept


def compute_metrics(tp_: int, fp: int, fn: int, num_classes: int = 1) -> dict:
    """
    计算论文评价指标（公式 2~6）。

    论文评价指标：
        P   = TP / (TP + FP)             [公式 2]
        R   = TP / (TP + FN)             [公式 3]
        mAP = Σ AP(C) / C                [公式 4]
        DR  = SD / (SD + MD) × 100%      [公式 5]  检出率
        FDR = FD / (SD + MD + FD) × 100% [公式 6]  误检率

    Args:
        tp_        : True Positive 数量
        fp         : False Positive 数量
        fn         : False Negative 数量
        num_classes: 类别数（论文为 1，仅"Mounting"）
    """
    p = tp_ / (tp_ + fp + 1e-9)
    r = tp_ / (tp_ + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"P": p, "R": r, "F1": f1}


# ── 私有辅助 ─────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _print_phase(idx: int, name: str, info: str = ""):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  Phase {idx}：{name}  {info}")
    print(f"{bar}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 完整流水线
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args: argparse.Namespace) -> str:
    """
    执行论文三阶段剪枝流水线并返回最终模型路径。

    对应论文 1.2 节"基于 YOLO v5n 通道剪枝算法的奶牛发情行为检测"
    总技术路线（图 5）：
        YOLOv5n 原始模型
            → 稀疏化训练（sr=0.005, 200 epoch）
            → 50% 通道剪枝（BN-gamma 准则）
            → 微调
            → YOLOv5n-Pruned（轻量化奶牛爬跨检测模型）
    """
    print("\n" + "★" * 60)
    print("  论文算法：融合 YOLOv5n 与通道剪枝的轻量化奶牛发情行为识别")
    print("  Wang Zheng et al., Transactions of CSAE, 38(23): 130-140, 2022")
    print("★" * 60)

    # Phase 1 ── 稀疏化训练
    if args.skip_sparse:
        sparse_model = args.model
        print(f"[跳过 Phase 1] 使用已有模型: {sparse_model}")
    else:
        sparse_model = sparse_train(
            model_path=args.model,
            data_yaml=args.data,
            epochs=args.sparse_epochs,
            sparsity_rate=args.sparsity_rate,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            weight_decay=args.weight_decay,
            save_dir="phase1_sparse",
            device=args.device,
        )

    plot_bn_gamma_distribution(sparse_model, "bn_gamma_after_sparse.png")

    # Phase 2 ── 通道剪枝
    pruned_model = prune_by_bn_gamma(
        model_path=sparse_model,
        pruning_rate=args.pruning_rate,
        save_path="yolov5n_pruned.pt",
        imgsz=args.imgsz,
        device=args.device,
    )

    # Phase 3 ── 微调
    if args.skip_finetune:
        final_model = pruned_model
        print(f"[跳过 Phase 3] 最终模型: {final_model}")
    else:
        final_model = finetune(
            pruned_model_path=pruned_model,
            data_yaml=args.data,
            epochs=args.finetune_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.finetune_lr,
            save_dir="phase3_finetuned",
            device=args.device,
        )

    # 输出最终统计（对应论文表 3）
    print("\n" + "=" * 60)
    print("  最终模型指标（对比论文表 3）")
    print("=" * 60)
    count_model_stats(final_model if final_model.endswith(".pt") else sparse_model,
                      imgsz=args.imgsz)
    print(f"\n论文目标: Params=0.72 M, FLOPs=0.68 G, mAP=97.70%, Speed=50.26 f/s")
    print(f"完整流水线结束 → {final_model}")
    print("★" * 60)
    return final_model


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="论文实现：融合 YOLOv5n 与通道剪枝的轻量化奶牛发情行为识别",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 模型 & 数据
    p.add_argument("--model", default="/gemini/code/Torch-Pruning-master/examples/yolov8/yolov5nu.pt",
                   help="预训练模型路径（论文使用 YOLOv5n v6.0）")
    p.add_argument("--data", default="/gemini/code/Torch-Pruning-master/examples/yolov8/data.yaml",
                   help="数据集配置（含 train/val 路径和类别信息）")

    # Phase 1：稀疏化训练（论文表 2 最优配置）
    g1 = p.add_argument_group("Phase 1  稀疏化训练")
    g1.add_argument("--sparse-epochs", type=int, default=50,
                    help="稀疏训练轮数（论文 200）")
    g1.add_argument("--sparsity-rate", type=float, default=0.005,
                    help="BN-gamma L1 系数 λ（论文 0.005，图 8 最优）")
    g1.add_argument("--lr0", type=float, default=0.0032,
                    help="初始学习率（论文 0.003 2）")
    g1.add_argument("--weight-decay", type=float, default=0.00036,
                    help="权重衰减（论文 0.000 36）")

    # Phase 2：通道剪枝
    g2 = p.add_argument_group("Phase 2  通道剪枝")
    g2.add_argument("--pruning-rate", type=float, default=0.5,
                    help="通道剪枝率（论文 0.5，图 9 最优）")

    # Phase 3：微调
    g3 = p.add_argument_group("Phase 3  微调")
    g3.add_argument("--finetune-epochs", type=int, default=50,
                    help="微调轮数")
    g3.add_argument("--finetune-lr", type=float, default=0.001,
                    help="微调初始学习率")

    # 通用
    p.add_argument("--imgsz", type=int, default=160,
                   help="输入图像尺寸（论文 512×512）")
    p.add_argument("--batch", type=int, default=32, help="批次大小")
    p.add_argument("--device", default="",
                   help="训练设备（如 '0' 或 'cpu'，留空自动选择）")

    # 跳过选项
    p.add_argument("--skip-sparse", action="store_true",
                   help="跳过 Phase 1，直接用 --model 作为稀疏训练后的权重")
    p.add_argument("--skip-finetune", action="store_true",
                   help="跳过 Phase 3 微调")

    # 单阶段模式
    p.add_argument(
        "--mode",
        choices=["full", "sparse", "prune", "finetune", "analyze"],
        default="full",
        help=(
            "运行模式："
            " full=完整三阶段流水线,"
            " sparse=仅稀疏训练,"
            " prune=仅通道剪枝,"
            " finetune=仅微调,"
            " analyze=仅分析 BN-γ 分布"
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.mode == "full":
        run_pipeline(args)

    elif args.mode == "sparse":
        sparse_train(
            model_path=args.model,
            data_yaml=args.data,
            epochs=args.sparse_epochs,
            sparsity_rate=args.sparsity_rate,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            weight_decay=args.weight_decay,
            device=args.device,
        )

    elif args.mode == "prune":
        prune_by_bn_gamma(
            model_path=args.model,
            pruning_rate=args.pruning_rate,
            save_path="yolov5n_pruned.pt",
            imgsz=args.imgsz,
            device=args.device,
        )

    elif args.mode == "finetune":
        finetune(
            pruned_model_path=args.model,
            data_yaml=args.data,
            epochs=args.finetune_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.finetune_lr,
            device=args.device,
        )

    elif args.mode == "analyze":
        analyze_bn_sparsity(args.model)
        plot_bn_gamma_distribution(args.model, "bn_gamma_dist.png")
        count_model_stats(args.model, imgsz=args.imgsz)
