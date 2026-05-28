"""
YOLO11 一键剪枝 + 蒸馏脚本
论文: A Lightweight Edge-Deployable Framework for Intelligent Rice Disease
      Monitoring Based on Pruning and Distillation (Wei Liu et al., Sensors 2026)

流程: 稀疏训练 → DepGraph 结构化剪枝 → CWD 知识蒸馏微调

直接运行:
    python prune_yolo11.py

修改下方 CONFIG 区域中的参数即可。
"""

# ============================================================
#  CONFIG — 按需修改这里
# ============================================================
CFG = dict(
    # ---------- 基础 ----------
    weights   = "/gemini/code/Torch-Pruning-master/examples/yolov8/yolo11n.pt",          # 初始权重（同时作为蒸馏教师）
    data      = "coco128.yaml",           # 数据集配置
    imgsz     = 256,
    batch     = 32,
    device    = "",                    # "" 自动选择，"0" 指定 GPU，"cpu" 强制 CPU
    # project   = "runs/prune",
    project   = "/gemini/output", # 使用离线训练时的保存路径

    # ---------- 稀疏训练（阶段 1） ----------
    sparse_epochs       = 50,
    lambda_sparse       = 1e-4,   # BN γ L1 正则化系数
    sparse_warmup_ratio = 0.10,   # 前 10% epoch 线性预热 lambda（防止初期振荡）
    sparse_smooth_eps   = 1e-3,   # 软化 sign 的 ε：w/(|w|+ε) 代替 sign(w)

    # ---------- DepGraph 剪枝（阶段 2） ----------
    K          = 3,                    # 分组数
    rhos       = (0.10, 0.30, 0.60),  # 各组剪枝率（高→低重要性）
    eps        = 4.0,                  # 重要性分数中的 ε（公式 7）
    min_keep   = 4,                    # 每层最少保留通道数

    # ---------- CWD 蒸馏微调（阶段 3） ----------
    distill_epochs      = 50,
    lambda_feat         = 1.5,    # 蒸馏损失权重（公式 11）
    distill_warmup_ratio= 0.20,   # 前 20% epoch 线性预热 lambda_feat（防止初期 loss 峰值）
    cwd_loss_clip       = 10.0,   # CWD loss 单步上限，防止特征差异大时爆炸
    T_d                 = 1.0,    # 温度系数
    # 学生/教师特征对齐层（YOLO11n neck 输出: P3/P4/P5）
    student_layers = [16, 19, 22],
    teacher_layers = [16, 19, 22],
)
# ============================================================

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_pruning as tp
    _TP_AVAILABLE = True
except ImportError:
    _TP_AVAILABLE = False

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


# ============================================================
#  §1  重要性评分 + 混合分组通道选择
# ============================================================

def bn_importance(module: nn.BatchNorm2d, eps: float = 4.0) -> torch.Tensor:
    """公式 7: I_c = |γ_c| × √(σ_c² + ε)"""
    return module.weight.data.abs() * torch.sqrt(module.running_var.data + eps)


def mixed_group_select(
    imp: torch.Tensor,
    K: int = 3,
    rhos: Tuple[float, ...] = (0.10, 0.30, 0.60),
    min_keep: int = 4,
) -> List[int]:
    """
    公式 8: 将通道按重要性降序排列后等分为 K 组，
    高重要性组剪枝率低（ρ=0.10），低重要性组剪枝率高（ρ=0.60）。
    返回需要剪掉的通道索引列表。
    """
    n = len(imp)
    if n <= min_keep:
        return []

    desc_idxs  = torch.argsort(imp, descending=True)
    group_size = max(1, n // K)
    prune_idxs: List[int] = []

    for k in range(K):
        g_start = k * group_size
        g_end   = (k + 1) * group_size if k < K - 1 else n
        g_idxs  = desc_idxs[g_start:g_end]
        if len(g_idxs) == 0:
            continue
        rho     = rhos[k] if k < len(rhos) else rhos[-1]
        n_prune = min(int(len(g_idxs) * rho), len(g_idxs) - min_keep)
        if n_prune <= 0:
            continue
        local_asc  = torch.argsort(imp[g_idxs])          # 组内升序 → 最不重要的在前
        prune_idxs += g_idxs[local_asc[:n_prune]].tolist()

    # 全局 min_keep 保护
    max_prune  = n - min_keep
    return prune_idxs[:max(0, max_prune)]


# ============================================================
#  §2  手动结构化剪枝（为 YOLO11n 定制，不依赖 torch_pruning 追踪）
#
#  YOLO11n 使用 Python list `y` 传递跳跃连接，导致 torch_pruning
#  无法追踪计算图。本实现直接读取 inner[i].f 属性解析依赖关系，
#  只剪顶层模块的"接口通道"（cv2 输出 / cv1 输入），
#  不触碰 C3k2/C2PSA 内部的残差/注意力连接。
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
        # C2PSA.cv2 是 ultralytics Conv wrapper
        return layer.cv2.conv, layer.cv2.bn
    # 通用回退：最后一个 Conv2d + BN
    convs = [m for m in layer.modules() if isinstance(m, nn.Conv2d)]
    bns   = [m for m in layer.modules() if isinstance(m, nn.BatchNorm2d)]
    return (convs[-1], bns[-1]) if convs and bns else (None, None)


def _get_in_conv(layer: nn.Module) -> Optional[nn.Conv2d]:
    """返回接收模块输入的第一个 nn.Conv2d。"""
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
    """穿透 Upsample，返回真正输出特征图的层索引。"""
    from ultralytics.nn.modules import Concat
    layer = inner[idx]
    if isinstance(layer, nn.Upsample):
        f = getattr(layer, 'f', -1)
        prev = idx - 1 if f == -1 else f
        return _resolve_real_source(prev, inner)
    return idx


def _concat_sources(concat_idx: int, inner, orig_ch: List[int]) -> List[Tuple[int, int]]:
    """
    对 Concat 层解析其来源：返回 [(real_layer_idx, channel_offset), ...]。
    orig_ch[i] 是原始（剪枝前）各层的输出通道数。
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


def depgraph_prune(
    model: nn.Module,
    imgsz: int = 640,
    K: int = 3,
    rhos: Tuple[float, ...] = (0.10, 0.30, 0.60),
    eps: float = 4.0,
    min_keep: int = 4,
    device: str = "cpu",
) -> nn.Module:
    """
    手动结构化剪枝，实现论文第 4.4 节混合分组归一化重要性规则（公式 7-8）。

    流程：
    1. 收集每层原始输出通道数
    2. 对每个可剪层计算重要性分数，用混合分组规则选出待剪通道
    3. 一次性裁剪所有层的输出通道
    4. 修复所有下游层的输入通道（含 Concat 偏移）
    """
    from ultralytics.nn.modules import Concat
    from ultralytics.nn.modules.head import Detect

    model = model.to(device).eval()
    inner = model.model  # nn.Sequential

    # ── 收集原始输出通道数 ──────────────────────────────
    orig_ch: List[int] = []
    for layer in inner:
        _, bn = _get_out_interface(layer)
        orig_ch.append(bn.num_features if bn is not None else 0)

    # ── 找出直接输入 Detect 头的层（输出通道不能剪）──────
    detect_sources: set = set()
    for layer in inner:
        if isinstance(layer, Detect):
            f = getattr(layer, 'f', [])
            if isinstance(f, list):
                for src in f:
                    detect_sources.add(_resolve_real_source(src, inner))
            break

    # ── Pass 1: 决定每层要剪的通道索引 ─────────────────
    prune_plan: Dict[int, List[int]] = {}   # layer_idx -> prune_idxs

    for i, layer in enumerate(inner):
        if i == 0:                              # stem 不剪
            continue
        if i in detect_sources:                 # 直接输入 Detect 的层不剪输出
            continue
        if isinstance(layer, (nn.Upsample, Concat, Detect)):
            continue

        conv_out, bn_out = _get_out_interface(layer)
        if conv_out is None or bn_out is None:
            continue

        imp  = bn_importance(bn_out, eps=eps)
        idxs = mixed_group_select(imp, K=K, rhos=rhos, min_keep=min_keep)
        if idxs:
            prune_plan[i] = idxs

    # ── Pass 2: 裁剪各层的输出通道 ──────────────────────
    for i, idxs in prune_plan.items():
        conv_out, bn_out = _get_out_interface(inner[i])
        keep = sorted(set(range(conv_out.out_channels)) - set(idxs))
        _prune_out_channels(conv_out, bn_out, keep)

    # ── Pass 3: 修复下游层的输入通道 ───────────────────
    # 对每个"真实接收层"（有输入卷积的层），解析其来源并修复 in_channels
    for j, layer in enumerate(inner):
        if isinstance(layer, (nn.Upsample, Concat, Detect)):
            continue

        in_conv = _get_in_conv(layer)
        if in_conv is None:
            continue

        f = getattr(layer, 'f', -1)

        # ── 情况 A：单一直接来源 ──
        if f == -1 or isinstance(f, int):
            prev_raw = j - 1 if f == -1 else f
            prev_real = _resolve_real_source(prev_raw, inner)
            if prev_real in prune_plan:
                _prune_in_channels(in_conv, prune_plan[prev_real])

        # ── 情况 B：经过 Concat 的多路来源 ──
        else:
            # f 是列表，说明 layer j 直接来自 Concat（f=[j-1, k] → Concat 已处理输出）
            # 实际上 YOLO11n 中有输入卷积的层不会直接有 list f，
            # list f 只用于 Concat/Detect。此分支用于泛化。
            sources = _concat_sources(j, inner, orig_ch)
            global_prune: List[int] = []
            for (src_real, offset) in sources:
                if src_real in prune_plan:
                    global_prune += [offset + x for x in prune_plan[src_real]]
            if global_prune:
                _prune_in_channels(in_conv, global_prune)

    # ── 处理 Concat 后的层（Concat 是多路输入汇合点）──
    # Concat 没有 in_conv，但其下游层的 in_conv 需要修复
    for j, layer in enumerate(inner):
        if isinstance(layer, (nn.Upsample, Concat, Detect)):
            continue

        in_conv = _get_in_conv(layer)
        if in_conv is None:
            continue

        f = getattr(layer, 'f', -1)
        prev_raw = j - 1 if f == -1 else (f if isinstance(f, int) else j - 1)
        prev_layer = inner[prev_raw]

        if not isinstance(prev_layer, Concat):
            continue

        # layer j 直接接在 Concat 后面 — 从 Concat 的来源追溯剪枝
        sources = _concat_sources(prev_raw, inner, orig_ch)
        global_prune: List[int] = []
        for (src_real, offset) in sources:
            if src_real in prune_plan:
                global_prune += [offset + x for x in prune_plan[src_real]]
        if global_prune:
            _prune_in_channels(in_conv, global_prune)

    params_after = sum(p.numel() for p in model.parameters())
    total_pruned = sum(len(v) for v in prune_plan.values())
    LOGGER.info(
        f"[剪枝] 完成: {len(prune_plan)} 个模块, 共 {total_pruned} 个通道 | "
        f"剩余参数: {params_after:,}"
    )
    return model


# ============================================================
#  §3  CWD 知识蒸馏
# ============================================================

class FeatureAligner(nn.Module):
    """1×1 卷积将学生特征通道数对齐到教师维度。"""
    def __init__(self, s_ch: int, t_ch: int):
        super().__init__()
        self.adapt = nn.Conv2d(s_ch, t_ch, 1, bias=False) if s_ch != t_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapt(x)


class CWDLoss(nn.Module):
    """
    通道级知识蒸馏损失（公式 9–11）。
    关键：φ_S 使用教师的归一化分母，而非学生自身（非对称 KL）。
    L_CWD = T_d² × Σ φ_T log(φ_T / φ_S)
    """
    def __init__(self, T_d: float = 1.0):
        super().__init__()
        self.T_d = T_d

    def forward(self, feat_t: torch.Tensor, feat_s: torch.Tensor) -> torch.Tensor:
        B, C   = feat_t.shape[:2]
        t_flat = feat_t.reshape(B, C, -1)
        s_flat = feat_s.reshape(B, C, -1)

        phi_T     = F.softmax(t_flat / self.T_d, dim=-1)
        log_Z_T   = torch.logsumexp(t_flat / self.T_d, dim=-1, keepdim=True)
        log_phi_S = s_flat / self.T_d - log_Z_T             # 学生用教师分母归一化

        kl = phi_T * (torch.log(phi_T.clamp(min=1e-8)) - log_phi_S)
        return (self.T_d ** 2) * kl.sum(dim=-1).mean()


# ============================================================
#  §4  稀疏训练器（阶段 1）
# ============================================================

class SparseTrainer(DetectionTrainer):
    """
    在 optimizer_step 中注入软化的 L1 稀疏梯度，推动低重要通道趋零。

    改进点（相比硬 sign 版本）:
    1. 软化 sign：w/(|w|+ε)  —— 权重趋近 0 时梯度也趋近 0，消除 0 附近的来回振荡
    2. Lambda 预热：前 warmup_ratio 的 epoch 内线性增大 lambda，避免初期剧烈扰动
    """

    lambda_sparse: float = 1e-4
    _sparse_warmup_ratio: float = 0.10
    _sparse_smooth_eps: float = 1e-3

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)

        # 线性预热：第 epoch 个 epoch 时的有效系数
        warmup_epochs = max(1, int(self.epochs * self._sparse_warmup_ratio))
        warmup_frac   = min(1.0, (self.epoch + 1) / warmup_epochs)
        eff_lambda    = self.lambda_sparse * warmup_frac

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) and m.weight.grad is not None:
                # 软化 sign：权重越小，施加的正则梯度越小，防止在 0 附近震荡
                soft_sign = m.weight.data / (m.weight.data.abs() + self._sparse_smooth_eps)
                m.weight.grad.data.add_(eff_lambda * soft_sign)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)  # 必须更新 EMA，否则验证用旧权重，mAP 永远为 0


# ============================================================
#  §5  蒸馏训练器（阶段 3）
# ============================================================

class DistillTrainer(DetectionTrainer):
    """
    在标准检测损失之上叠加 CWD 蒸馏损失。

    改进点（相比原版）:
    1. Lambda 预热：前 warmup_ratio 的 epoch 内线性增大 lambda_feat，
       避免初期学生/教师特征差异大时 CWD loss 过大冲击检测损失
    2. CWD loss 截断：单步 CWD loss 超过 cwd_loss_clip 时截断，防止异常峰值
    """

    def __init__(self, *args,
                 teacher_weights: str,
                 student_layers: List[int],
                 teacher_layers: List[int],
                 lambda_feat: float = 1.5,
                 distill_warmup_ratio: float = 0.20,
                 cwd_loss_clip: float = 10.0,
                 T_d: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher_weights       = teacher_weights
        self._student_layers        = student_layers
        self._teacher_layers        = teacher_layers
        self._lambda_feat           = lambda_feat
        self._distill_warmup_ratio  = distill_warmup_ratio
        self._cwd_loss_clip         = cwd_loss_clip
        self._T_d                   = T_d
        self._teacher: Optional[nn.Module] = None
        self._aligners: Optional[nn.ModuleList] = None
        self._cwd              = CWDLoss(T_d=T_d)
        self._criterion_ready  = False   # 只包装一次 criterion
        self._cur_imgs: Optional[torch.Tensor] = None

    # ── 教师模型 ───────────────────────────────────────
    def _load_teacher(self):
        t = YOLO(self._teacher_weights).model.to(self.device).eval()
        for p in t.parameters():
            p.requires_grad_(False)
        return t

    # ── 特征提取（hook 方式，不做额外 forward）─────────
    def _get_feats(
        self,
        model: nn.Module,
        layer_idxs: List[int],
        imgs: torch.Tensor,
        no_grad: bool = False,
    ) -> List[torch.Tensor]:
        feats: Dict[int, torch.Tensor] = {}
        inner = model.model if hasattr(model, "model") else model
        hooks = [
            inner[i].register_forward_hook(
                lambda _, __, o, i=i: feats.__setitem__(i, o)
            )
            for i in layer_idxs
        ]
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            model(imgs)
        for h in hooks:
            h.remove()
        return [feats[i] for i in layer_idxs]

    def _ensure_aligners(self, s_feats, t_feats):
        if self._aligners is not None:
            return
        self._aligners = nn.ModuleList([
            FeatureAligner(sf.shape[1], tf.shape[1]).to(self.device)
            for sf, tf in zip(s_feats, t_feats)
        ])

    def _distill_loss(self, imgs: torch.Tensor) -> torch.Tensor:
        if self._teacher is None:
            self._teacher = self._load_teacher()

        t_feats = self._get_feats(self._teacher, self._teacher_layers, imgs, no_grad=True)
        s_feats = self._get_feats(self.model,    self._student_layers, imgs, no_grad=False)
        self._ensure_aligners(s_feats, t_feats)

        loss = torch.tensor(0.0, device=self.device)
        for aligner, sf, tf in zip(self._aligners, s_feats, t_feats):
            sf = aligner(sf)
            if sf.shape[2:] != tf.shape[2:]:
                sf = F.interpolate(sf, size=tf.shape[2:], mode="bilinear", align_corners=False)
            loss = loss + self._cwd(tf, sf)
        return loss / max(len(self._student_layers), 1)

    # ── criterion 懒初始化 + 一次性包装 ───────────────
    def _setup_cwd_criterion(self):
        """在模型和 criterion 都就绪后只调用一次。"""
        if self._criterion_ready:
            return
        # 确保 criterion 已初始化（不同 ultralytics 版本行为不同）
        if not hasattr(self.model, 'criterion') or self.model.criterion is None:
            self.model.criterion = self.model.init_criterion()
        base = self.model.criterion
        trainer_ref = self

        class _CWDWrapper:
            def __call__(self_, preds, batch_):
                base_loss, items = base(preds, batch_)
                if trainer_ref._cur_imgs is not None:
                    cwd = trainer_ref._distill_loss(trainer_ref._cur_imgs)
                    # 截断：防止初期特征差异极大时 CWD loss 爆炸
                    cwd = cwd.clamp(max=trainer_ref._cwd_loss_clip)
                    # 预热：前 warmup_ratio 的 epoch 线性增大有效系数
                    warmup_epochs = max(1, int(trainer_ref.epochs * trainer_ref._distill_warmup_ratio))
                    warmup_frac   = min(1.0, (trainer_ref.epoch + 1) / warmup_epochs)
                    eff_lambda    = trainer_ref._lambda_feat * warmup_frac
                    base_loss = base_loss + eff_lambda * cwd
                return base_loss, items

        self.model.criterion = _CWDWrapper()
        self._criterion_ready = True

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        self._cur_imgs = batch["img"]          # 供 _CWDWrapper 使用
        self._setup_cwd_criterion()            # 只在第一次调用时包装
        return batch


# ============================================================
#  §6  三阶段流水线
# ============================================================

def stage1_sparse(cfg: dict) -> str:
    """稀疏训练：让低重要通道 BN 权重趋近于零。"""
    LOGGER.info("\n" + "="*60)
    LOGGER.info("  阶段 1/3：稀疏训练")
    LOGGER.info("="*60)

    SparseTrainer.lambda_sparse        = cfg["lambda_sparse"]
    SparseTrainer._sparse_warmup_ratio = cfg["sparse_warmup_ratio"]
    SparseTrainer._sparse_smooth_eps   = cfg["sparse_smooth_eps"]
    trainer = SparseTrainer(overrides=dict(
        model    = cfg["weights"],
        data     = cfg["data"],
        epochs   = cfg["sparse_epochs"],
        imgsz    = cfg["imgsz"],
        batch    = cfg["batch"],
        device   = cfg["device"],
        project  = cfg["project"],
        name     = "1_sparse",
        cos_lr   = True,          # 余弦 LR 调度，曲线更平滑
        exist_ok = True,
    ))
    trainer.train()
    best = str(trainer.best)
    LOGGER.info(f"[阶段1] 完成 → {best}")
    return best


def stage2_prune(cfg: dict, sparse_weights: str) -> str:
    """DepGraph 一次性结构化剪枝。"""
    LOGGER.info("\n" + "="*60)
    LOGGER.info("  阶段 2/3：DepGraph 结构化剪枝")
    LOGGER.info("="*60)

    if not _TP_AVAILABLE:
        raise ImportError("请先安装依赖: pip install torch-pruning")

    device = cfg["device"] if cfg["device"] else "cpu"

    yolo_obj  = YOLO(sparse_weights)
    nn_model  = copy.deepcopy(yolo_obj.model)
    params_before = sum(p.numel() for p in nn_model.parameters())
    LOGGER.info(f"[阶段2] 剪枝前参数量: {params_before:,}")

    nn_model = depgraph_prune(
        nn_model,
        imgsz    = cfg["imgsz"],
        K        = cfg["K"],
        rhos     = cfg["rhos"],
        eps      = cfg["eps"],
        min_keep = cfg["min_keep"],
        device   = device,
    )

    params_after = sum(p.numel() for p in nn_model.parameters())
    ratio = 1.0 - params_after / params_before
    LOGGER.info(f"[阶段2] 剪枝后参数量: {params_after:,}  (压缩 {ratio*100:.1f}%)")

    out_dir  = Path(cfg["project"]) / "2_pruned"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / "pruned.pt")

    torch.save(
        {"model": nn_model, "nc": nn_model.nc, "names": nn_model.names, "imgsz": cfg["imgsz"]},
        out_path,
    )
    LOGGER.info(f"[阶段2] 剪枝后模型已保存 → {out_path}")
    return out_path


def stage3_distill(cfg: dict, pruned_weights: str) -> str:
    """CWD 知识蒸馏微调。"""
    LOGGER.info("\n" + "="*60)
    LOGGER.info("  阶段 3/3：CWD 知识蒸馏微调")
    LOGGER.info("="*60)

    trainer = DistillTrainer(
        overrides=dict(
            model    = pruned_weights,
            data     = cfg["data"],
            epochs   = cfg["distill_epochs"],
            imgsz    = cfg["imgsz"],
            batch    = cfg["batch"],
            device   = cfg["device"],
            project  = cfg["project"],
            name     = "3_distill",
            cos_lr   = True,          # 余弦 LR 调度，曲线更平滑
            exist_ok = True,
        ),
        teacher_weights       = cfg["weights"],   # 原始 yolo11n.pt 作为教师
        student_layers        = cfg["student_layers"],
        teacher_layers        = cfg["teacher_layers"],
        lambda_feat           = cfg["lambda_feat"],
        distill_warmup_ratio  = cfg["distill_warmup_ratio"],
        cwd_loss_clip         = cfg["cwd_loss_clip"],
        T_d                   = cfg["T_d"],
    )
    trainer.train()
    best = str(trainer.best)
    LOGGER.info(f"[阶段3] 完成 → {best}")
    return best


# ============================================================
#  主函数
# ============================================================

def main():
    LOGGER.info("\n" + "="*60)
    LOGGER.info("  YOLO11 剪枝 + 蒸馏一键流水线")
    LOGGER.info(f"  初始权重  : {CFG['weights']}")
    LOGGER.info(f"  数据集    : {CFG['data']}")
    LOGGER.info(f"  输出目录  : {CFG['project']}")
    LOGGER.info("="*60)

    # 阶段 1：稀疏训练
    sparse_weights = stage1_sparse(CFG)

    # 阶段 2：DepGraph 剪枝
    pruned_weights = stage2_prune(CFG, sparse_weights)

    # 阶段 3：CWD 蒸馏微调
    final_weights  = stage3_distill(CFG, pruned_weights)

    LOGGER.info("\n" + "="*60)
    LOGGER.info("  全流程完成！")
    LOGGER.info(f"  最终权重  : {final_weights}")
    LOGGER.info("="*60)


if __name__ == "__main__":
    main()
