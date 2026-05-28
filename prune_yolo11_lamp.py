"""
YOLO11 LAMP 通道剪枝 + 微调流水线
论文: Sugarcane stem node detection with algorithm based on improved YOLO11
     channel pruning with small target enhancement (Wen et al., PLOS One 2025)

算法: LAMP — Layer-Adaptive Magnitude-Based Pruning（公式 2-16）
      Score(i; W) = (W[i])² / Σ_{j≥i} (W[j])²
      W[i] = 第 i 个输出通道卷积核的 ℓ1-范数（升序排列）

流程: LAMP 通道剪枝（无需稀疏预训练）→ 标准检测微调

直接运行:
    python prune_yolo11_lamp.py
"""

# ============================================================
#  CONFIG — 按需修改这里
# ============================================================
CFG = dict(
    # ---------- 基础 ----------
    weights   = "yolo11n.pt",        # 初始权重
    data      = "data.yaml",         # 数据集配置
    imgsz     = 288,
    batch     = 32,
    device    = "",                  # "" 自动选择，"0"=GPU，"cpu"=强制 CPU
    project   = "runs/prune_lamp",

    # ---------- LAMP 剪枝参数 ----------
    # 论文原始参数: speed_up=3.0, steps=200, reg=5e-4, reg_inc=1e-5
    # 对标准 YOLO11n 建议先用 speed_up=2.0（保留 50% 通道）
    speed_up  = 2.0,   # 目标压缩比：保留 1/speed_up 的通道
    steps     = 200,   # 迭代步数（线性渐进剪枝）
    reg       = 5e-4,  # 初始正则化强度
    reg_inc   = 1e-5,  # 每步正则化增量
    min_keep  = 4,     # 每层最少保留通道数

    # ---------- 微调参数 ----------
    finetune_epochs =50,
)
# ============================================================

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


# ============================================================
#  §1  LAMP 通道重要性打分
# ============================================================

def lamp_channel_importance(conv: nn.Conv2d) -> torch.Tensor:
    """
    LAMP score per output channel (Eq. 2-16, Wen et al. 2025).

    步骤:
    1. 计算每个输出通道滤波器的 ℓ1-范数
    2. 升序排列后计算后缀平方和
    3. Score(i) = norm_i² / Σ_{j≥i} norm_j²  （越大越重要）
    """
    # [C_out, C_in, kH, kW] → ℓ1-norm per output channel
    norms = conv.weight.data.abs().sum(dim=[1, 2, 3])   # [C_out]

    # 升序排列（小→大）
    sorted_norms, sort_idx = torch.sort(norms)
    sq = sorted_norms ** 2

    # 后缀平方和: suffix_sum[i] = Σ_{j≥i} sq[j]
    suffix_sum = torch.flip(
        torch.cumsum(torch.flip(sq, [0]), dim=0), [0]
    )

    # LAMP scores for sorted positions
    lamp_sorted = sq / (suffix_sum + 1e-12)

    # 映射回原始通道顺序
    scores = torch.zeros_like(norms)
    scores[sort_idx] = lamp_sorted
    return scores


def lamp_select(
    scores: torch.Tensor,
    keep_n: int,
) -> List[int]:
    """返回需要剪掉的通道索引（保留得分最高的 keep_n 个通道）。"""
    n = len(scores)
    n_prune = n - keep_n
    if n_prune <= 0:
        return []
    # 升序 → 分数最低的在前 → 剪掉前 n_prune 个
    asc_idx = torch.argsort(scores)
    return asc_idx[:n_prune].tolist()


# ============================================================
#  §2  YOLO11n 结构感知辅助函数
#
#  YOLO11n 使用 Python list `y` 传递跳跃连接，torch_pruning 无法追踪。
#  本实现直接读取 inner[i].f 属性解析依赖关系，
#  只操作顶层模块的"接口通道"，不触碰内部残差。
# ============================================================

def _get_out_interface(layer: nn.Module):
    """返回 (output_conv: nn.Conv2d, output_bn: nn.BatchNorm2d)。"""
    from ultralytics.nn.modules import Conv as UConv
    from ultralytics.nn.modules.block import C3k2, SPPF, C2PSA
    if isinstance(layer, UConv):
        return layer.conv, layer.bn
    if isinstance(layer, (C3k2, SPPF)):
        return layer.cv2.conv, layer.cv2.bn
    if isinstance(layer, C2PSA):
        return layer.cv2.conv, layer.cv2.bn
    convs = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]
    bns   = [m for m in layer.modules() if isinstance(m, nn.BatchNorm2d)]
    return (convs[-1], bns[-1]) if convs and bns else (None, None)


def _get_in_conv(layer: nn.Module) -> Optional[nn.Conv2d]:
    """返回接收输入的第一个 nn.Conv2d。"""
    from ultralytics.nn.modules import Conv as UConv
    from ultralytics.nn.modules.block import C3k2, SPPF, C2PSA
    if isinstance(layer, UConv):
        return layer.conv
    if isinstance(layer, (C3k2, SPPF)):
        return layer.cv1.conv
    if isinstance(layer, C2PSA):
        return layer.cv1.conv
    for m in layer.modules():
        if isinstance(m, nn.Conv2d):
            return m
    return None


def _resolve_real_source(idx: int, inner) -> int:
    """穿透 Upsample 层，返回真正产生特征的层索引。"""
    layer = inner[idx]
    if isinstance(layer, nn.Upsample):
        f = getattr(layer, 'f', -1)
        prev = idx - 1 if f == -1 else f
        return _resolve_real_source(prev, inner)
    return idx


def _concat_sources(concat_idx: int, inner, orig_ch: List[int]) -> List[Tuple[int, int]]:
    """
    对 Concat 层解析来源，返回 [(real_layer_idx, channel_offset), ...]。
    orig_ch[i] 为剪枝前各层输出通道数。
    """
    f = getattr(inner[concat_idx], 'f', [])
    raw = [concat_idx - 1 if x == -1 else x for x in f]
    result, offset = [], 0
    for s in raw:
        real = _resolve_real_source(s, inner)
        result.append((real, offset))
        offset += orig_ch[real] or 0
    return result


def _prune_out_channels(conv: nn.Conv2d, bn: nn.BatchNorm2d, keep: List[int]):
    conv.weight       = nn.Parameter(conv.weight.data[keep].contiguous())
    if conv.bias is not None:
        conv.bias     = nn.Parameter(conv.bias.data[keep])
    conv.out_channels = len(keep)
    bn.weight         = nn.Parameter(bn.weight.data[keep])
    bn.bias           = nn.Parameter(bn.bias.data[keep])
    bn.running_mean   = bn.running_mean[keep]
    bn.running_var    = bn.running_var[keep]
    bn.num_features   = len(keep)


def _prune_in_channels(conv: nn.Conv2d, global_prune_idxs: List[int]):
    keep = sorted(set(range(conv.in_channels)) - set(global_prune_idxs))
    if len(keep) == conv.in_channels:
        return
    conv.weight      = nn.Parameter(conv.weight.data[:, keep].contiguous())
    conv.in_channels = len(keep)


# ============================================================
#  §3  LAMP 结构化剪枝主函数
# ============================================================

def lamp_prune(
    model: nn.Module,
    speed_up: float = 2.0,
    steps: int = 200,
    reg: float = 5e-4,
    reg_inc: float = 1e-5,
    min_keep: int = 4,
    device: str = "cpu",
) -> nn.Module:
    """
    LAMP 通道剪枝（Wen et al. 2025 §2 节）。

    算法（逐步线性递增剪枝）:
    - 共 steps 步，第 t 步目标保留率 = 1 - (t/steps)×(1 - 1/speed_up)
    - 每步对各层计算 LAMP score，选出待剪通道（取最终步结果应用）
    - 正则化强度 reg_t = reg + t×reg_inc（影响 LAMP 分数扰动）
    - 一次性应用最终剪枝方案（不修改中间权重）
    - 修复所有下游输入通道（含 Concat 偏移）

    参数:
        speed_up : 目标加速比。论文使用 3.0（参数量降至 1/3），
                   标准 YOLO11n 建议 2.0（保留 50% 通道）。
        steps    : 迭代步数，线性插值最终剪枝率，论文 200。
        reg      : 初始正则化系数，论文 5e-4。
        reg_inc  : 每步增量，论文 1e-5。
        min_keep : 每层最少保留通道数。
    """
    from ultralytics.nn.modules import Concat
    from ultralytics.nn.modules.head import Detect

    model = model.to(device).eval()
    inner = model.model   # nn.Sequential

    # ── 收集原始输出通道数 ──────────────────────────────
    orig_ch: List[int] = []
    for layer in inner:
        _, bn = _get_out_interface(layer)
        orig_ch.append(bn.num_features if bn is not None else 0)

    # ── 找出直接输入 Detect 的层（保护其输出通道不被剪）──
    detect_sources: set = set()
    for layer in inner:
        if isinstance(layer, Detect):
            f = getattr(layer, 'f', [])
            if isinstance(f, list):
                for src in f:
                    detect_sources.add(_resolve_real_source(src, inner))
            break

    # ── 候选剪枝层 ──────────────────────────────────────
    candidate: List[int] = []
    for i, layer in enumerate(inner):
        if i == 0 or i in detect_sources:
            continue
        if isinstance(layer, (nn.Upsample, Concat, Detect)):
            continue
        conv_out, bn_out = _get_out_interface(layer)
        if conv_out is not None and bn_out is not None:
            candidate.append(i)

    # ── 迭代 LAMP 剪枝（200 步线性递进）────────────────
    # 第 t 步目标保留通道数（线性从 n 降到 n/speed_up）
    prune_plan: Dict[int, List[int]] = {}

    for step in range(1, steps + 1):
        frac   = step / steps
        # 当前步目标保留率
        keep_r = 1.0 - frac * (1.0 - 1.0 / speed_up)
        # 正则化强度（影响 LAMP score 的扰动因子，模拟权重衰减效果）
        reg_t  = reg + (step - 1) * reg_inc

        for i in candidate:
            conv_out, _ = _get_out_interface(inner[i])
            n = orig_ch[i]    # 使用原始通道数保持一致
            keep_n = max(min_keep, round(n * keep_r))

            # 计算 LAMP score（在原始权重上，加小扰动模拟正则化）
            scores = lamp_channel_importance(conv_out)
            # 正则化扰动：将低分通道的分数进一步压低（无需梯度）
            scores = scores - reg_t * (scores < scores.mean()).float()

            idxs = lamp_select(scores, keep_n)
            if idxs:
                prune_plan[i] = idxs
            else:
                prune_plan.pop(i, None)

    # ── Pass 2: 裁剪输出通道 ────────────────────────────
    for i, idxs in prune_plan.items():
        conv_out, bn_out = _get_out_interface(inner[i])
        keep = sorted(set(range(conv_out.out_channels)) - set(idxs))
        _prune_out_channels(conv_out, bn_out, keep)

    # ── Pass 3a: 修复直接前驱的输入通道 ─────────────────
    for j, layer in enumerate(inner):
        if isinstance(layer, (nn.Upsample, Concat, Detect)):
            continue
        in_conv = _get_in_conv(layer)
        if in_conv is None:
            continue
        f = getattr(layer, 'f', -1)
        if f == -1 or isinstance(f, int):
            prev_raw  = j - 1 if f == -1 else f
            prev_real = _resolve_real_source(prev_raw, inner)
            if prev_real in prune_plan:
                _prune_in_channels(in_conv, prune_plan[prev_real])

    # ── Pass 3b: 修复 Concat 后继层的输入通道 ──────────
    for j, layer in enumerate(inner):
        if isinstance(layer, (nn.Upsample, Concat, Detect)):
            continue
        in_conv = _get_in_conv(layer)
        if in_conv is None:
            continue
        f = getattr(layer, 'f', -1)
        prev_raw  = j - 1 if f == -1 else (f if isinstance(f, int) else j - 1)
        prev_layer = inner[prev_raw]
        if not isinstance(prev_layer, Concat):
            continue

        sources = _concat_sources(prev_raw, inner, orig_ch)
        global_prune: List[int] = []
        for (src_real, offset) in sources:
            if src_real in prune_plan:
                global_prune += [offset + x for x in prune_plan[src_real]]
        if global_prune:
            _prune_in_channels(in_conv, global_prune)

    params_after  = sum(p.numel() for p in model.parameters())
    total_pruned  = sum(len(v) for v in prune_plan.values())
    LOGGER.info(
        f"[LAMP 剪枝] 完成: {len(prune_plan)} 个模块, 共 {total_pruned} 个通道 | "
        f"剩余参数: {params_after:,}"
    )
    return model


# ============================================================
#  §4  两阶段流水线
# ============================================================

def stage1_lamp_prune(cfg: dict) -> str:
    """LAMP 通道剪枝（无需稀疏训练）。"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("  阶段 1/2：LAMP 通道剪枝")
    LOGGER.info("=" * 60)

    device = cfg["device"] if cfg["device"] else "cpu"

    yolo_obj = YOLO(cfg["weights"])
    nn_model = copy.deepcopy(yolo_obj.model)
    p0 = sum(p.numel() for p in nn_model.parameters())
    LOGGER.info(f"[阶段1] 剪枝前参数量: {p0:,}")

    nn_model = lamp_prune(
        nn_model,
        speed_up = cfg["speed_up"],
        steps    = cfg["steps"],
        reg      = cfg["reg"],
        reg_inc  = cfg["reg_inc"],
        min_keep = cfg["min_keep"],
        device   = device,
    )

    p1    = sum(p.numel() for p in nn_model.parameters())
    ratio = 1.0 - p1 / p0
    LOGGER.info(f"[阶段1] 剪枝后参数量: {p1:,}  (压缩 {ratio*100:.1f}%，实际加速比 {p0/p1:.2f}x)")

    out_dir  = Path(cfg["project"]) / "1_pruned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "pruned.pt")

    torch.save(
        {
            "model" : nn_model,
            "nc"    : nn_model.nc,
            "names" : nn_model.names,
            "imgsz" : cfg["imgsz"],
        },
        out_path,
    )
    LOGGER.info(f"[阶段1] 剪枝模型已保存 → {out_path}")
    return out_path


def stage2_finetune(cfg: dict, pruned_weights: str) -> str:
    """标准检测训练对剪枝后模型进行微调。"""
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("  阶段 2/2：检测微调")
    LOGGER.info("=" * 60)

    trainer = DetectionTrainer(overrides=dict(
        model    = pruned_weights,
        data     = cfg["data"],
        epochs   = cfg["finetune_epochs"],
        imgsz    = cfg["imgsz"],
        batch    = cfg["batch"],
        device   = cfg["device"],
        project  = cfg["project"],
        name     = "2_finetune",
        exist_ok = True,
    ))
    trainer.train()
    best = str(trainer.best)
    LOGGER.info(f"[阶段2] 完成 → {best}")
    return best


# ============================================================
#  主函数
# ============================================================

def main():
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("  YOLO11 LAMP 通道剪枝 + 微调流水线")
    LOGGER.info(f"  初始权重  : {CFG['weights']}")
    LOGGER.info(f"  数据集    : {CFG['data']}")
    LOGGER.info(f"  目标压缩  : {CFG['speed_up']}x  (迭代 {CFG['steps']} 步)")
    LOGGER.info(f"  输出目录  : {CFG['project']}")
    LOGGER.info("=" * 60)

    # 阶段 1：LAMP 剪枝
    pruned_weights = stage1_lamp_prune(CFG)

    # 阶段 2：微调
    final_weights = stage2_finetune(CFG, pruned_weights)

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("  全流程完成！")
    LOGGER.info(f"  最终权重  : {final_weights}")
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
