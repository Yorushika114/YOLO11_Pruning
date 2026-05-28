#!/usr/bin/env python3
"""
train_yolov5s_mgct.py — YOLOv5s-MGCT 训练脚本

论文: 改进YOLOv5s的轨道障碍物检测模型轻量化研究 (李昂, 2023)
期刊: 计算机工程与应用, 59(4), 197-207

四项改进 (MGCT):
  M  Mixup 批量数据增强 — Beta(0.2, 0.2), 替代 Mosaic (论文 Section 2.1)
  G  GhostConv / C3Ghost backbone + neck (在 yolov5s_mgct.yaml 中声明)
  C  Coordinate Attention CoordAtt (在 yolov5s_mgct.yaml 中声明)
  T  稀疏训练 (BN γ 的 L1 正则, λ=0.002) + 通道剪枝 (60%) + 微调

工作流 (默认一键完成)
--------------------
完整流程 — 稀疏训练 → 自动剪枝 → 微调 (推荐):
    python train_yolov5s_mgct.py --data data.yaml --epochs 300 --imgsz 608

仅做稀疏训练, 不自动剪枝:
    python train_yolov5s_mgct.py --data data.yaml --stage1_only

从已有权重直接执行剪枝 + 微调:
    python train_yolov5s_mgct.py --data data.yaml \\
        --prune --weights runs/detect/mgct/weights/best.pt --prune_ratio 0.6

论文实验结果 (表5) — 最佳剪枝率 60%:
    mAP 0.5% = 93.9, 模型体积 = 4.7 MB, FPS = 95
    (原始 YOLOv5s: mAP=93.7, 14.4MB, 82 FPS)
"""

from __future__ import annotations

import argparse
import copy
import math
import re
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


# =============================================================================
# M  Mixup 批量数据增强
# 论文 Section 2.1, 公式 (1)-(3):
#   λ = Beta(α, β)
#   mixed_x = λ * x1 + (1-λ) * x2
#   mixed_y = λ * y1 + (1-λ) * y2
# 检测任务适配: 图像线性混合; 边界框取两张图的并集 (所有框都保留).
# =============================================================================

def mixup_batch(batch: Dict, alpha: float = 0.2) -> Dict:
    """
    对一个训练 batch 执行 Mixup 数据增强 (论文 Section 2.1).

    从 Beta(alpha, alpha) 采样混合系数 λ, 将每张图与 batch 内随机另一张图混合.
    图像做线性混合; 两张图的所有边界框取并集 (检测任务标准做法).

    Args:
        batch: ultralytics 训练 batch 字典, 含 img / cls / bboxes / batch_idx.
        alpha: Beta 分布参数 (论文使用 0.2).

    Returns:
        原地修改后的 batch 字典.
    """
    img = batch["img"]          # (B, 3, H, W), 已归一化, 已在 device 上
    B = img.size(0)
    if B < 2:
        return batch

    lam = float(np.random.beta(alpha, alpha))           # 公式 (1)
    perm = torch.randperm(B, device=img.device)

    # 图像线性混合  公式 (2)
    batch["img"] = lam * img + (1.0 - lam) * img[perm]

    # 边界框并集: 对 image i, 保留自身框 + perm[i] 的框 (设 batch_idx=i)
    cls   = batch["cls"]         # (N, 1)
    boxes = batch["bboxes"]      # (N, 4) xywh 归一化
    bidx  = batch["batch_idx"]   # (N,)

    extra_cls, extra_boxes, extra_bidx = [], [], []
    for i in range(B):
        partner = perm[i].item()
        if partner == i:
            continue
        mask = (bidx == partner)
        if not mask.any():
            continue
        extra_cls.append(cls[mask])
        extra_boxes.append(boxes[mask])
        extra_bidx.append(
            torch.full((int(mask.sum()),), i, dtype=bidx.dtype, device=bidx.device)
        )

    if extra_cls:
        batch["cls"]       = torch.cat([cls]   + extra_cls,   dim=0)
        batch["bboxes"]    = torch.cat([boxes] + extra_boxes, dim=0)
        batch["batch_idx"] = torch.cat([bidx]  + extra_bidx,  dim=0)

    return batch


# =============================================================================
# T  稀疏训练 — BN 缩放因子 γ 的 L1 正则化
# 论文 Section 2.4:
#   对每个通道引入缩放因子 γ 并与该通道相乘, |γ| 表示通道重要性.
#   将 L1 范数添加到 γ 的梯度使不重要通道的 γ 趋近于零, 便于后续剪枝.
#   论文设置: sparsity λ = 0.002, 训练批次 = 300
# =============================================================================

def apply_sparse_grad(model: nn.Module, lambda_s: float) -> None:
    """
    向所有 BatchNorm2d 的缩放参数 γ 添加 L1 惩罚梯度.

    在 optimizer.step() 之前调用 (梯度已 unscale), 公式等效于:
        grad_γ += λ * sign(γ)

    Args:
        model:    正在训练的检测模型.
        lambda_s: L1 系数 (论文: 0.002).
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and m.weight.grad is not None:
            m.weight.grad.data.add_(lambda_s * torch.sign(m.weight.data))


# =============================================================================
# T  通道剪枝
# 论文 Section 2.4 & 表5:
#   剪枝率 60% 时精度损失最小 (mAP 仅降 0.2pp), 速度/体积提升最大.
#   流程: 稀疏训练 → 按 |γ| 排序 → 置零最小 ratio 份额的通道 → 微调.
# =============================================================================

def _collect_bn_gammas(model: nn.Module) -> torch.Tensor:
    """收集模型所有 BatchNorm2d 的 |γ| 并拼成一维向量."""
    return torch.cat(
        [m.weight.data.abs().clone() for m in model.modules()
         if isinstance(m, nn.BatchNorm2d)]
    )


def print_gamma_stats(model: nn.Module, tag: str = "") -> None:
    """打印 BN γ 分布统计 (调试/监控用)."""
    g = _collect_bn_gammas(model)
    LOGGER.info(
        f"[BN γ 统计 {tag}]  数量={g.numel()}  "
        f"均值={g.mean():.4f}  中位={g.median():.4f}  "
        f"|γ|<0.01 占比={float((g < 0.01).sum()) / g.numel():.1%}"
    )


def prune_model(model: nn.Module, ratio: float = 0.6) -> int:
    """
    软剪枝: 将 |γ| 最小的 ratio 比例通道的 BN 权重置零.

    论文公式 (10): W2 = argmin Loss(W1)
    置零 BN 的 weight(γ) 与 bias(β), 相当于把该通道输出强制为零.
    这是"软剪枝" — 模型结构不变但通道被禁用.
    如需物理减小模型尺寸, 还需额外做 Conv+BN 层的通道重参数化 (本脚本不含).

    论文表5 最优设置:
        ratio=0.60 → mAP 0.5%=93.9, 体积=4.7MB, FPS=95

    Args:
        model: 经过稀疏训练的模型.
        ratio: 全局通道剪枝率, 0~1 之间 (论文推荐 0.6).

    Returns:
        被剪枝的通道数量.
    """
    all_w = _collect_bn_gammas(model)
    if all_w.numel() == 0:
        LOGGER.warning("[Prune] 未找到 BatchNorm2d 层, 跳过剪枝.")
        return 0

    k = max(1, int(ratio * all_w.numel()))
    threshold = float(torch.kthvalue(all_w, k).values)

    pruned = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            mask = m.weight.data.abs() <= threshold
            m.weight.data[mask] = 0.0
            m.bias.data[mask]   = 0.0
            pruned += int(mask.sum())

    LOGGER.info(
        f"[Prune] 已剪枝 {pruned}/{all_w.numel()} 个 BN 通道  "
        f"(ratio={ratio:.0%}, 阈值={threshold:.6f})"
    )
    return pruned


# =============================================================================
# 自定义 Trainer
# =============================================================================

# 模块级配置字典 — 在 train() 调用前设置, 避免将非标准 key 传入 ultralytics overrides
_mgct_cfg: Dict = {
    "mixup_alpha":   0.2,    # Mixup Beta(α,α) 参数
    "sparse_lambda": 0.002,  # BN γ 稀疏正则系数
    "use_sparse":    True,   # 是否启用稀疏训练
}


class MGCTTrainer(DetectionTrainer):
    """
    YOLOv5s-MGCT 训练器, 在标准 DetectionTrainer 基础上添加:
      - M: preprocess_batch 中注入 Mixup 批量增强
      - T: optimizer_step 中在 unscale 后注入 L1 稀疏梯度
    """

    # ── M: Mixup ──────────────────────────────────────────────────────────────
    def preprocess_batch(self, batch: Dict) -> Dict:
        """标准预处理 (移到 GPU + 归一化) 后执行 Mixup."""
        batch = super().preprocess_batch(batch)      # img → device, /255
        return mixup_batch(batch, _mgct_cfg["mixup_alpha"])

    # ── T: 稀疏梯度 ───────────────────────────────────────────────────────────
    def optimizer_step(self) -> None:
        """
        梯度 unscale → 注入稀疏 L1 梯度 → 梯度裁剪 → optimizer.step → EMA.

        覆盖父类方法在 unscale_ 与 step 之间插入 L1 正则梯度,
        确保与 AMP (Automatic Mixed Precision) 正确配合.
        """
        self.scaler.unscale_(self.optimizer)
        if _mgct_cfg["use_sparse"]:
            apply_sparse_grad(self.model, _mgct_cfg["sparse_lambda"])
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)


# =============================================================================
# CLI 参数解析
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv5s-MGCT 训练脚本 (李昂 2023)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型 / 数据 ────────────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default="/gemini/code/Torch-Pruning-master/examples/yolov8/yolov5nu.pt",
        help="模型 YAML 路径 (阶段1) 或 .pt 权重路径",
    )
    p.add_argument(
        "--weights",
        default="",
        help="预训练 .pt 权重; 若同时指定 --prune 则对这批权重执行剪枝",
    )
    p.add_argument(
        "--data",
        default="/gemini/code/Torch-Pruning-master/examples/yolov8/data.yaml",
        help="数据集 YAML",
    )

    # ── 训练超参 ────────────────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=50,
                   help="训练轮数 (论文: 300)")
    p.add_argument("--imgsz",         type=int,   default=160,
                   help="输入分辨率 (论文: 608×608)")
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--device",        default="",
                   help="CUDA 设备, 如 0 / 0,1 / cpu")
    p.add_argument("--workers",       type=int,   default=16)
    p.add_argument("--project",       default="runs/detect")
    p.add_argument("--name",          default="mgct")
    p.add_argument("--lr0",           type=float, default=0.01,
                   help="初始学习率")
    p.add_argument("--lrf",           type=float, default=0.01,
                   help="最终学习率倍率 (lr_final = lr0 * lrf)")
    p.add_argument("--momentum",      type=float, default=0.937)
    p.add_argument("--weight_decay",  type=float, default=5e-4)

    # ── M: Mixup ────────────────────────────────────────────────────────────────
    p.add_argument("--mixup_alpha", type=float, default=0.2,
                   help="Mixup 的 Beta(α,α) 参数 (论文图2)")

    # ── T: 稀疏训练 ────────────────────────────────────────────────────────────
    p.add_argument("--sparse",    dest="sparse", action="store_true",  default=True,
                   help="启用 BN γ 的 L1 稀疏正则 (论文 Section 2.4, 默认开启)")
    p.add_argument("--no_sparse", dest="sparse", action="store_false",
                   help="关闭稀疏训练")
    p.add_argument("--sparse_lambda", type=float, default=0.002,
                   help="L1 正则系数 (论文: 0.002)")

    # ── T: 剪枝 ────────────────────────────────────────────────────────────────
    p.add_argument("--prune",        action="store_true",
                   help="跳过稀疏训练, 直接对 --weights 执行剪枝+微调 (需同时指定 --weights)")
    p.add_argument("--prune_ratio",  type=float, default=0.6,
                   help="全局通道剪枝率 (论文最优: 0.6)")
    p.add_argument("--finetune_epochs", type=int, default=None,
                   help="微调轮数, 默认与 --epochs 相同 (论文: 300)")
    p.add_argument("--stage1_only",  action="store_true",
                   help="只做稀疏训练, 不自动执行剪枝微调")

    return p.parse_args()


# =============================================================================
# 两阶段训练
# =============================================================================

def _base_train_kwargs(args: argparse.Namespace) -> Dict:
    """生成 yolo.train() 的公共参数字典."""
    return dict(
        data         = args.data,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        device       = args.device,
        workers      = args.workers,
        project      = args.project,
        mosaic       = 0.0,   # 禁用 Mosaic (论文: 用 Mixup 替代)
        mixup        = 0.0,   # 关闭内置 Mixup, 由 MGCTTrainer.preprocess_batch 处理
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
    )


def stage1(args: argparse.Namespace) -> Path:
    """
    阶段1: 稀疏训练 (M + G + C + T-sparse).

    - Mosaic 关闭, 改用批量 Mixup (MGCTTrainer.preprocess_batch)
    - 稀疏正则通过 MGCTTrainer.optimizer_step 注入 BN γ 梯度
    - G/C 改进已在 yolov5s_mgct.yaml 中声明, 无需额外操作

    Returns:
        Path: 最优检查点 best.pt 的路径 (供阶段2使用).
    """
    LOGGER.info("\n" + "=" * 65)
    LOGGER.info(
        f"  阶段 1: YOLOv5s-MGCT 稀疏训练\n"
        f"  Mixup alpha={args.mixup_alpha}  "
        f"sparse={'开' if args.sparse else '关'}  "
        f"lambda={args.sparse_lambda}"
    )
    LOGGER.info("=" * 65 + "\n")

    # 将参数写入模块级配置 (MGCTTrainer 通过 _mgct_cfg 读取)
    _mgct_cfg["mixup_alpha"]   = args.mixup_alpha
    _mgct_cfg["sparse_lambda"] = args.sparse_lambda
    _mgct_cfg["use_sparse"]    = args.sparse

    model_path = args.weights if args.weights else args.model
    yolo = YOLO(model_path)

    yolo.train(
        trainer = MGCTTrainer,
        name    = args.name,
        lr0     = args.lr0,
        lrf     = args.lrf,
        **_base_train_kwargs(args),
    )

    # 打印训练结束后的 γ 分布 (稀疏程度统计)
    if args.sparse:
        print_gamma_stats(yolo.model, tag="稀疏训练后")

    # 定位 best.pt — 优先从 trainer 属性读取, 否则按约定路径构建
    best_pt: Path | None = None
    trainer = getattr(yolo, "trainer", None)
    if trainer is not None:
        candidate = getattr(trainer, "best", None)
        if candidate and Path(candidate).exists():
            best_pt = Path(candidate)

    if best_pt is None:
        best_pt = Path(args.project) / args.name / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(
            f"[Stage1] 找不到最优权重文件: {best_pt}\n"
            "请确认训练已正常完成, 或手动用 --prune --weights <path> 进入阶段2."
        )

    LOGGER.info(f"[Stage1] 最优权重: {best_pt}")
    return best_pt


def stage2(args: argparse.Namespace, weights_path: str | Path | None = None) -> None:
    """
    阶段2: 通道剪枝 + 微调 (T-prune).

    流程 (论文 Section 2.4 & 图7):
      1. 载入权重 (优先使用 weights_path, 其次 args.weights)
      2. 打印 γ 分布统计
      3. 软剪枝: 置零 |γ| 最小的 prune_ratio 比例通道的 BN 权重
      4. 将剪枝后权重保存为新 .pt 文件
      5. 用较小学习率微调剪枝模型

    Args:
        args:         命令行参数.
        weights_path: 阶段1返回的 best.pt 路径 (由 pipeline 自动传入).
                      若为 None 则使用 args.weights.
    """
    src_pt = str(weights_path or args.weights)
    if not src_pt:
        raise ValueError("需要指定权重文件: 通过 --weights 或由阶段1自动传入.")
    if not Path(src_pt).exists():
        raise FileNotFoundError(f"权重文件不存在: {src_pt}")

    LOGGER.info("\n" + "=" * 65)
    LOGGER.info(
        f"  阶段 2: 通道剪枝 ({args.prune_ratio:.0%}) + 微调\n"
        f"  权重来源: {src_pt}"
    )
    LOGGER.info("=" * 65 + "\n")

    # ── 1. 载入权重 ───────────────────────────────────────────────────────────
    yolo = YOLO(src_pt)
    print_gamma_stats(yolo.model, tag="剪枝前")

    # ── 2. 软剪枝 ─────────────────────────────────────────────────────────────
    n_pruned = prune_model(yolo.model, ratio=args.prune_ratio)
    print_gamma_stats(yolo.model, tag="剪枝后")

    # ── 3. 将剪枝权重另存 (供断点续训 / 部署) ────────────────────────────────
    save_dir = Path(args.project) / (args.name + "_pruned")
    save_dir.mkdir(parents=True, exist_ok=True)
    pruned_pt = save_dir / "pruned.pt"

    # 复用原始 checkpoint 格式, 只替换 model 权重
    orig_ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)
    orig_ckpt["model"]        = copy.deepcopy(yolo.model).half()
    orig_ckpt["epoch"]        = -1
    orig_ckpt["best_fitness"] = None
    orig_ckpt["optimizer"]    = None
    torch.save(orig_ckpt, pruned_pt)
    LOGGER.info(f"[Prune] 剪枝权重已保存 → {pruned_pt}")

    # ── 4. 微调剪枝模型 ───────────────────────────────────────────────────────
    # 关闭稀疏正则; 仍保留 Mixup 增强; LR 降为预训练的 1/10
    _mgct_cfg["mixup_alpha"]   = args.mixup_alpha
    _mgct_cfg["sparse_lambda"] = 0.0
    _mgct_cfg["use_sparse"]    = False

    ft_epochs = args.finetune_epochs if args.finetune_epochs is not None else args.epochs
    ft_kwargs = _base_train_kwargs(args)
    ft_kwargs["epochs"] = ft_epochs          # 微调轮数可单独设置

    yolo_ft = YOLO(str(pruned_pt))
    yolo_ft.train(
        trainer = MGCTTrainer,
        name    = args.name + "_finetune",
        lr0     = args.lr0 * 0.1,      # 微调 LR = 预训练 LR / 10 (论文)
        lrf     = args.lrf,
        **ft_kwargs,
    )

    LOGGER.info(
        f"\n阶段2 完成.  已剪枝通道数: {n_pruned}\n"
        f"微调权重位于: {args.project}/{args.name}_finetune/weights/"
    )


# =============================================================================
# 入口
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.prune:
        # 仅阶段2: 对已有权重执行剪枝 + 微调
        stage2(args)

    elif args.stage1_only:
        # 仅阶段1: 稀疏训练, 不自动剪枝
        stage1(args)

    else:
        # 默认完整流程: 稀疏训练 → 自动剪枝 → 微调 (一步到位)
        LOGGER.info("\n[Pipeline] 完整 MGCT 流程: 稀疏训练 → 剪枝 → 微调\n")
        best_pt = stage1(args)
        stage2(args, weights_path=best_pt)

    LOGGER.info("\n✓ YOLOv5s-MGCT 训练完成.")


if __name__ == "__main__":
    main()
