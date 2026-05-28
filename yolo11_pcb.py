"""
yolo11_pcb.py
=============
基于 YOLO11 的轻量化 PCB 缺陷检测算法（单文件实现）

论文: 基于YOLO11的轻量化PCB缺陷检测算法研究
      黄文杰, 罗维平 等 | 广西师范大学学报 2026, 44(1): 56-67

三项核心改进:
  1. BiMAFPN (BiFPNFusion) — 双向加权特征融合，替换 Neck 中 Fusion 3/4 的 Concat
  2. C3k2_Faster           — FasterBlock 替换 Bottleneck，部分卷积降低冗余计算
  3. LSCD                  — 共享卷积检测头 + GroupNorm + Scale，减少 34.6% 参数

用法:
  训练:   python yolo11_pcb.py train --data pcb_defect.yaml --epochs 200
  验证:   python yolo11_pcb.py val   --data pcb_defect.yaml --weights runs/.../best.pt
  推理:   python yolo11_pcb.py predict --source images/ --weights runs/.../best.pt
  信息:   python yolo11_pcb.py info
"""

# ─────────────────────────────────────────────────────────────────────────────
# 标准库
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import ast
import contextlib
import math
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch
# ─────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# ultralytics 基础导入（自定义模块依赖）
# ─────────────────────────────────────────────────────────────────────────────
from ultralytics.nn.modules.block import C2f, DFL
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.ops import make_divisible
from ultralytics.utils import LOGGER, colorstr

# ─────────────────────────────────────────────────────────────────────────────
# 所有 ultralytics 模块导入到当前命名空间
# （parse_model 使用 globals()[module_name] 查找模块，
#   必须在本文件命名空间中可见）
# ─────────────────────────────────────────────────────────────────────────────
from ultralytics.nn.modules import (
    # block.py
    C1, C2, C2PSA, C3, C3TR, CIB, DFL, ELAN1, PSA, SPP, SPPELAN, SPPF,
    A2C2f, AConv, ADown, Attention, BNContrastiveHead, Bottleneck, BottleneckCSP,
    C2f, C2fAttn, C2fCIB, C2fPSA, C3Ghost, C3k2, C3x, CBFuse, CBLinear,
    ContrastiveHead, CoordAtt, GhostBottleneck, HGBlock, HGStem, ImagePoolingAttn,
    MaxSigmoidAttnBlock, Proto, RepC3, RepNCSPELAN4, RepVGGDW, ResNetLayer,
    SCDown, TorchVision, my_conv,
    # conv.py
    CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv,
    DWConvTranspose2d, Focus, GhostConv, Index, LightConv, RepConv, SpatialAttention,
    # head.py
    OBB, Classify, Detect, LRPCHead, Pose, RTDETRDecoder, Segment, WorldDetect,
    YOLOEDetect, YOLOESegment, v10Detect,
    # transformer.py
    AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer,
    LayerNorm2d, MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer,
    TransformerLayer,
)


# ═════════════════════════════════════════════════════════════════════════════
#  第一部分: 自定义模块定义
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
#  改进 1 — C3k2_Faster
#  PConv → FasterBlock → C3k2_Faster
# ─────────────────────────────────────────────

class PConv(nn.Module):
    """
    部分卷积 (Partial Convolution)。
    仅对前 1/n_div 个通道执行 3×3 卷积，其余通道直接跳过，
    避免在高度相似的特征图管道中进行大量冗余计算。
    """

    def __init__(self, c: int, k: int = 3, n_div: int = 4):
        super().__init__()
        self.dim_conv = c // n_div
        self.partial_conv = nn.Conv2d(
            self.dim_conv, self.dim_conv, k, 1, autopad(k), bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, : self.dim_conv], x[:, self.dim_conv:]
        return torch.cat([self.partial_conv(x1), x2], dim=1)


class FasterBlock(nn.Module):
    """
    FasterNet Block（论文图 3d）。
    结构: PConv 3×3 → Conv S=1,K=3 → Conv 1×1 + 残差连接。
    部分卷积负责空间混合，两个逐点/小卷积负责通道混合。
    """

    def __init__(self, c: int, n_div: int = 4):
        super().__init__()
        self.pconv = PConv(c, k=3, n_div=n_div)
        self.cv1 = Conv(c, c, 3)            # Conv S=1, K=3
        self.cv2 = Conv(c, c, 1, act=False)  # Conv 1×1，无激活

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(self.pconv(x)))


class C3k2_Faster(C2f):
    """
    C3k2_Faster: 将 C3k2 中的 Bottleneck 替换为 FasterBlock。
    与 C3k2 接口完全兼容（可在 YAML 中直接替换）。
    在保持精度的同时降低 FLOPs 和参数量。
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(FasterBlock(self.c) for _ in range(n))


# ─────────────────────────────────────────────
#  辅助模块 — ConvGn（GroupNorm 卷积，用于 LSCD）
# ─────────────────────────────────────────────

class ConvGn(nn.Module):
    """
    Conv + GroupNorm + SiLU。
    GroupNorm 将通道分组归一化，对小 batch 和多尺度训练
    更稳定（相比 BatchNorm 不依赖 batch 统计量）。
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, num_groups: int = 0):
        super().__init__()
        g = num_groups if num_groups > 0 else max(1, c2 // 16)
        while c2 % g != 0 and g > 1:
            g //= 2
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), bias=False)
        self.gn = nn.GroupNorm(g, c2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


# ─────────────────────────────────────────────
#  改进 2 — BiFPNFusion
#  替换 Neck 中 Fusion 3、4 位置的 Concat
# ─────────────────────────────────────────────

class BiFPNFusion(nn.Module):
    """
    BiFPN 双向加权特征融合，替换 PANet 中的 Concat 操作（论文第 2.1 节）。

    公式 (6)(7):
        P_out = Conv( Σ(ω_i · Resize(P_i)) / (Σω_i + ε) )

    与 Concat 相比:
      - 加权求和而非拼接，输出通道数不翻倍
      - 可学习权重 (ReLU 归一化) 让模型自动决定各尺度重要性
      - 通过 1×1 投影对齐通道，3×3 卷积精炼融合特征

    Args:
        c_list: 各输入特征图的通道数列表
        c_out:  融合后输出通道数
    """

    def __init__(self, c_list: list, c_out: int):
        super().__init__()
        self.eps = 1e-4
        self.weights = nn.Parameter(
            torch.ones(len(c_list), dtype=torch.float32))
        self.projs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            for c in c_list
        )
        self.conv = Conv(c_out, c_out, 3)

    def forward(self, x: list) -> torch.Tensor:
        target_size = x[0].shape[2:]   # x[0] 决定目标空间分辨率
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        fused = sum(
            w[i] * F.interpolate(self.projs[i](x[i]),
                                 size=target_size, mode="nearest")
            for i in range(len(x))
        )
        return self.conv(fused)


# ─────────────────────────────────────────────
#  改进 3 — LSCD
#  轻量化共享卷积检测头
# ─────────────────────────────────────────────

class _Scale(nn.Module):
    """可学习标量缩放层，为每个检测尺度提供独立的幅度自适应调节。"""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LSCD(Detect):
    """
    Lightweight Shared Convolutional Detection head（论文第 2.3 节 / 图 5）。

    结构:
      ① 各尺度 ConvGn 1×1  — 通道对齐与信息交换
      ② 共享 ConvGn 3×3    — 同一权重应用于 P3/P4/P5，1 份参数代替 3 份
      ③ 各尺度 Conv_Reg → Scale  — bbox 回归输出
         各尺度 Conv_Cls → Scale  — 分类输出

    参数量减少来源:
      · GroupNorm 替代 BatchNorm（无额外运行参数）
      · 3×3 ConvGn 权重跨尺度共享（1×参数 vs 3×参数）
      · Scale 层仅含 1 个可学习标量，开销极小
    """

    def __init__(self, nc: int = 80, ch: tuple = ()):
        # 跳过 Detect.__init__，直接用 nn.Module 初始化
        nn.Module.__init__(self)

        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        # 取中间尺度（P4）通道数作为共享维度
        mid_ch = ch[self.nl // 2]
        c_reg = max(16, mid_ch // 4, self.reg_max * 4)

        # ① 各尺度 1×1 对齐（轻量，参数量极少）
        self.proj = nn.ModuleList(ConvGn(c, mid_ch, 1) for c in ch)

        # ② 共享 3×3 卷积体（核心参数节省处）
        self.shared_reg_conv = ConvGn(mid_ch, c_reg, 3)
        self.shared_cls_conv = ConvGn(mid_ch, mid_ch, 3)

        # ③ 各尺度输出层: Conv2d → Scale
        #    cv2/cv3[i][0] = Conv2d, cv2/cv3[i][1] = _Scale
        self.cv2 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c_reg, 4 * self.reg_max, 1), _Scale())
            for _ in range(self.nl)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(mid_ch, self.nc, 1), _Scale())
            for _ in range(self.nl)
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: list):
        # ① 投影到 mid_ch
        feats = [self.proj[i](x[i]) for i in range(self.nl)]

        # ② 共享卷积（相同权重，各尺度独立空间尺寸）
        reg_feats = [self.shared_reg_conv(f) for f in feats]
        cls_feats = [self.shared_cls_conv(f) for f in feats]

        # ③ 合并回归与分类输出
        out = []
        for i in range(self.nl):
            reg = self.cv2[i](reg_feats[i])
            cls = self.cv3[i](cls_feats[i])
            out.append(torch.cat([reg, cls], dim=1))

        if self.training:
            return out
        y = self._inference(out)
        return y if self.export else (y, out)

    def bias_init(self):
        """初始化检测偏置。cv2/cv3[i][0] 是各自的 Conv2d。"""
        for cv2_i, cv3_i, s in zip(self.cv2, self.cv3, self.stride):
            cv2_i[0].bias.data[:] = 1.0
            cv3_i[0].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / s) ** 2)


# ═════════════════════════════════════════════════════════════════════════════
#  第二部分: 替换 parse_model（注入自定义模块支持）
# ═════════════════════════════════════════════════════════════════════════════
# globals() 在此函数内部返回本文件的全局命名空间，
# 所有 ultralytics 模块已在顶部 import，自定义模块也已定义，
# 因此 globals()[module_name_string] 可以找到所有模块。

def parse_model(d: dict, ch: int, verbose: bool = True):
    """
    解析 YOLO 模型 YAML 字典为 PyTorch 模型（支持 PCB 自定义模块）。
    本函数是 ultralytics.nn.tasks.parse_model 的替换版本，
    额外支持: C3k2_Faster, BiFPNFusion, LSCD。
    """
    import ast
    import contextlib

    # ── 基础参数 ──
    legacy = True
    max_channels = float("inf")
    scale = ""

    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (
        d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
    )

    if scales:
        scale = d.get("scale", "")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"未指定 model scale，使用默认 scale='{scale}'。")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
        )

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]

    # ── 模块分组（决定 args 如何处理）──
    base_modules = frozenset({
        Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck,
        SPP, SPPF, C2fPSA, C2PSA, DWConv, Focus, BottleneckCSP, C1, C2, C2f,
        C3k2, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn, C3, C3TR,
        C3Ghost, CoordAtt, torch.nn.ConvTranspose2d, DWConvTranspose2d, C3x,
        RepC3, PSA, SCDown, C2fCIB, A2C2f, my_conv,
        C3k2_Faster,   # ← PCB 新增
    })

    repeat_modules = frozenset({
        BottleneckCSP, C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost, C3x,
        RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        C3k2_Faster,   # ← PCB 新增
    })

    # ── 逐层解析 ──
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]   # 在本文件命名空间中查找
        )
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals(
                    ) else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(
                    min(args[1], max_channels // 2) * width, 8)
                args[2] = int(
                    max(round(
                        min(args[2], max_channels // 2 // 32)) * width, 1)
                    if args[2] > 1
                    else args[2]
                )
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m in {C3k2, C3k2_Faster}:    # M/L/X 尺度下启用 c3k
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False

        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {
            Detect, WorldDetect, YOLOEDetect, Segment, YOLOESegment,
            Pose, OBB, ImagePoolingAttn, v10Detect,
            LSCD,   # ← PCB 新增
        }:
            args.append([ch[x] for x in f])
            if m in {Segment, YOLOESegment}:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, YOLOESegment, Pose, OBB, LSCD}:
                m.legacy = legacy
            c2 = nc
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in {TorchVision, Index}:
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        elif m is BiFPNFusion:              # ← PCB 新增
            c_in_list = [ch[x] for x in f]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c_in_list, c2]
        else:
            c2 = ch[f]

        m_ = (
            torch.nn.Sequential(*(m(*args)
                                for _ in range(n))) if n > 1 else m(*args)
        )
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}"
            )
        save.extend(x % i for x in (
            [f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)


def patch_ultralytics():
    """
    将自定义模块注入 ultralytics 框架，替换 parse_model。
    必须在创建任何 YOLO 模型实例之前调用。
    """
    import ultralytics.nn.tasks as tasks

    # 注册到 tasks 命名空间（供框架其他部分使用）
    tasks.C3k2_Faster = C3k2_Faster
    tasks.BiFPNFusion = BiFPNFusion
    tasks.LSCD = LSCD

    # 替换 parse_model
    tasks.parse_model = parse_model

    LOGGER.info("✓ PCB 模块已注册: C3k2_Faster | BiFPNFusion | LSCD")


# ═════════════════════════════════════════════════════════════════════════════
#  第三部分: 模型 YAML 配置
# ═════════════════════════════════════════════════════════════════════════════

# nc 设为 80（训练时会根据数据集自动更新）
YOLO11N_PCB_YAML = """\
# yolo11n_pcb.yaml
# 基于 YOLO11n 的轻量化 PCB 缺陷检测模型
# 改进: BiMAFPN (BiFPNFusion@Fusion3/4) + C3k2_Faster + LSCD

nc: 80
scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00,  512]
  l: [1.00, 1.00,  512]
  x: [1.00, 1.50,  512]

# ── Backbone（与 YOLO11n 完全一致）──
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv,  [64,   3, 2]]          # 0  P1/2
  - [-1, 1, Conv,  [128,  3, 2]]          # 1  P2/4
  - [-1, 2, C3k2,  [256,  False, 0.25]]   # 2
  - [-1, 1, Conv,  [256,  3, 2]]          # 3  P3/8
  - [-1, 2, C3k2,  [512,  False, 0.25]]   # 4
  - [-1, 1, Conv,  [512,  3, 2]]          # 5  P4/16
  - [-1, 2, C3k2,  [512,  True]]          # 6
  - [-1, 1, Conv,  [1024, 3, 2]]          # 7  P5/32
  - [-1, 2, C3k2,  [1024, True]]          # 8
  - [-1, 1, SPPF,  [1024, 5]]             # 9
  - [-1, 2, C2PSA, [1024]]                # 10

# ── Head（Neck + 检测头）──
# Fusion 1/2: 保留原始 Concat（自顶向下路径）
# Fusion 3/4: BiFPNFusion 替换 Concat（自底向上路径）
# C3k2 → C3k2_Faster（Neck 全部替换）
# 检测头 Detect → LSCD
head:
  - [-1,        1, nn.Upsample,  [None, 2, "nearest"]]  # 11
  - [[-1, 6],   1, Concat,       [1]]                    # 12  Fusion 1
  - [-1,        2, C3k2_Faster,  [512,  False]]          # 13

  - [-1,        1, nn.Upsample,  [None, 2, "nearest"]]  # 14
  - [[-1, 4],   1, Concat,       [1]]                    # 15  Fusion 2
  - [-1,        2, C3k2_Faster,  [256,  False]]          # 16  P3/8-small

  - [-1,        1, Conv,         [256, 3, 2]]            # 17
  - [[-1, 13],  1, BiFPNFusion,  [512]]                  # 18  Fusion 3 (BiFPN)
  - [-1,        2, C3k2_Faster,  [512,  False]]          # 19  P4/16-medium

  - [-1,        1, Conv,         [512, 3, 2]]            # 20
  - [[-1, 10],  1, BiFPNFusion,  [1024]]                 # 21  Fusion 4 (BiFPN)
  - [-1,        2, C3k2_Faster,  [1024, True]]           # 22  P5/32-large

  - [[16, 19, 22], 1, LSCD, [nc]]                        # 23  LSCD 检测头
"""


_VALID_SCALES = ("n", "s", "m", "l", "x")


def get_model_yaml_path(scale: str = "n") -> str:
    """
    将内置 YAML 配置写入脚本同级目录下的 yolo11{scale}_pcb.yaml 并返回路径。
    文件名中的 scale 字符使 ultralytics 自动识别对应尺度（n/s/m/l/x）。

    Args:
        scale: 模型尺度，'n' | 's' | 'm' | 'l' | 'x'
    """
    scale = scale.lower()
    if scale not in _VALID_SCALES:
        raise ValueError(f"scale 须为 {_VALID_SCALES}，当前: {scale!r}")
    yaml_path = Path(__file__).parent / f"yolo11{scale}_pcb.yaml"
    yaml_path.write_text(YOLO11N_PCB_YAML, encoding="utf-8")
    return str(yaml_path)


# ═════════════════════════════════════════════════════════════════════════════
#  第四部分: 训练 / 验证 / 推理接口
# ═════════════════════════════════════════════════════════════════════════════

def build_model(model: str = "n"):
    """
    构建 PCB 检测模型。

    Args:
        model: 三种写法均可
               · 'n'|'s'|'m'|'l'|'x'   → 使用内置 PCB 架构对应尺度
               · 'path/to/arch.yaml'    → 使用自定义架构 YAML
               · 'path/to/weights.pt'   → 加载已有权重（含架构）
    Returns:
        ultralytics YOLO 实例
    """
    patch_ultralytics()
    from ultralytics import YOLO

    m = str(model).strip()
    suffix = Path(m).suffix.lower()

    if suffix == ".pt":
        yolo = YOLO(m)
        LOGGER.info(f"已加载权重: {m}")
    elif suffix in (".yaml", ".yml"):
        yolo = YOLO(str(Path(m).resolve()))
        LOGGER.info(f"已加载自定义配置: {m}")
    else:
        yaml_path = get_model_yaml_path(m)   # 校验 n/s/m/l/x
        yolo = YOLO(yaml_path)
        LOGGER.info(f"已构建 yolo11{m}-pcb 模型: {yaml_path}")

    return yolo


def train(
    data: str,
    epochs: int = 200,
    batch: int = 32,
    imgsz: int = 640,
    device: str = "",
    lr0: float = 0.01,
    lrf: float = 0.0001,
    optimizer: str = "SGD",
    model: str = "n",
    cfg: str = None,
    project: str = "runs/pcb",
    name: str = "train",
    **kwargs,
):
    """
    训练 PCB 缺陷检测模型。

    对应论文训练设置:
      输入 640×640 | 200 epoch | Batch 32 | SGD lr0=0.01→0.0001

    Args:
        data:      数据集 YAML 路径（含 train/val/test 路径及类别定义）
        epochs:    训练轮数
        batch:     每批样本数
        imgsz:     输入图像尺寸
        device:    训练设备，如 '0', '0,1', 'cpu'
        lr0:       初始学习率（cfg 未指定时生效）
        lrf:       最终学习率绝对值（cfg 未指定时生效）
        optimizer: 优化器（cfg 未指定时生效）
        model:     见 build_model(): n/s/m/l/x | *.yaml | *.pt
        cfg:       训练配置 YAML 路径（ultralytics 超参数文件，
                   指定后 lr0/lrf/optimizer 等被文件中的值覆盖）
        project:   结果保存根目录
        name:      本次实验子目录名
    """
    yolo = build_model(model)

    train_kwargs = dict(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        **kwargs,
    )
    if cfg:
        # 由配置文件接管超参数，不再单独传 lr0/lrf/optimizer
        train_kwargs["cfg"] = cfg
    else:
        train_kwargs.update(
            lr0=lr0,
            lrf=lrf / lr0,   # ultralytics lrf 是相对值
            optimizer=optimizer,
        )

    results = yolo.train(**train_kwargs)
    return results


def val(
    data: str,
    weights: str,
    imgsz: int = 640,
    batch: int = 32,
    device: str = "",
    **kwargs,
):
    """验证模型精度，输出 mAP@0.5 和 mAP@0.5:0.95。"""
    patch_ultralytics()
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.val(
        data=data, imgsz=imgsz, batch=batch, device=device, **kwargs
    )
    return results


def predict(
    source,
    weights: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "",
    save: bool = True,
    **kwargs,
):
    """对图像/视频/目录运行 PCB 缺陷推理。"""
    patch_ultralytics()
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=save,
        **kwargs,
    )
    return results


def show_info(model: str = "n"):
    """打印模型结构与参数统计。model 同 build_model()，支持尺度/yaml/pt。"""
    build_model(model).info(detailed=False)


# ═════════════════════════════════════════════════════════════════════════════
#  第五部分: 命令行入口
# ═════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于YOLO11的轻量化PCB缺陷检测算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
--model 三种写法（train / info 通用）:
  n/s/m/l/x          → 内置PCB架构，默认 n
  path/to/arch.yaml  → 自定义架构配置文件
  path/to/best.pt    → 加载已有权重（微调/继续训练）

示例:
  # 从头训练，内置 n-scale
  python yolo11_pcb.py train --data pcb_defect.yaml

  # 使用更大的 s-scale 架构
  python yolo11_pcb.py train --data pcb_defect.yaml --model s

  # 用自定义架构 YAML
  python yolo11_pcb.py train --data pcb_defect.yaml --model my_arch.yaml

  # 加载已有权重做微调
  python yolo11_pcb.py train --data pcb_defect.yaml --model runs/pcb/train/weights/best.pt

  # 验证
  python yolo11_pcb.py val --data pcb_defect.yaml --weights runs/pcb/train/weights/best.pt

  # 推理
  python yolo11_pcb.py predict --source test_images/ --weights runs/pcb/train/weights/best.pt

  # 查看内置 m-scale 结构
  python yolo11_pcb.py info --model m

  # 查看自定义配置结构
  python yolo11_pcb.py info --model my_arch.yaml
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── 公共模型选择参数（train / info 共用）──
    def _add_model_args(p, default_scale="n"):
        p.add_argument(
            "--model",
            type=str,
            default=default_scale,
            metavar="MODEL",
            help=(
                "模型来源，三种写法:\n"
                "  n/s/m/l/x          → 内置PCB架构对应尺度（默认 n）\n"
                "  path/to/arch.yaml  → 自定义架构配置文件\n"
                "  path/to/best.pt    → 已有权重（含架构，用于微调）"
            ),
        )

    # ── train ──
    p_train = sub.add_parser("train", help="训练模型")
    p_train.add_argument("--data",   default="coco128.yaml",    required=True,
                         help="数据集 YAML 路径")
    p_train.add_argument("--epochs",     type=int,
                         default=200, help="训练轮数，默认 200")
    p_train.add_argument("--batch",      type=int,
                         default=32,  help="Batch size，默认 32")
    p_train.add_argument("--imgsz",      type=int,
                         default=640, help="输入图像尺寸，默认 640")
    p_train.add_argument("--device",     type=str,
                         default="",  help="训练设备: '0' / '0,1' / 'cpu'")
    p_train.add_argument("--lr0",        type=float,
                         default=0.01, help="初始学习率，默认 0.01")
    p_train.add_argument("--lrf",        type=float,
                         default=1e-4, help="最终学习率绝对值，默认 1e-4")
    p_train.add_argument("--optimizer",  type=str,   default="SGD",
                         choices=["SGD", "Adam", "AdamW", "RMSProp"],
                         help="优化器，默认 SGD")
    p_train.add_argument("--cfg",         type=str,   default=None,
                         metavar="YAML",
                         help="训练配置文件路径（ultralytics 超参数 YAML，"
                              "指定后 --lr0/--lrf/--optimizer 均被文件覆盖）")
    p_train.add_argument("--project",    type=str,
                         default="runs/pcb", help="结果保存根目录")
    p_train.add_argument("--name",       type=str,
                         default="train",    help="实验子目录名")
    _add_model_args(p_train)

    # ── val ──
    p_val = sub.add_parser("val", help="验证模型精度")
    p_val.add_argument("--data",    required=True,
                       help="数据集 YAML 路径")
    p_val.add_argument("--weights", required=True,
                       help="模型权重 .pt 路径")
    p_val.add_argument("--imgsz",   type=int,   default=640)
    p_val.add_argument("--batch",   type=int,   default=32)
    p_val.add_argument("--device",  type=str,   default="")

    # ── predict ──
    p_pred = sub.add_parser("predict", help="运行推理")
    p_pred.add_argument("--source",  required=True,
                        help="图像/目录/视频路径")
    p_pred.add_argument("--weights", required=True,
                        help="模型权重 .pt 路径")
    p_pred.add_argument("--imgsz",   type=int,   default=640)
    p_pred.add_argument("--conf",    type=float, default=0.25, help="置信度阈值")
    p_pred.add_argument("--iou",     type=float,
                        default=0.45, help="NMS IoU 阈值")
    p_pred.add_argument("--device",  type=str,   default="")
    p_pred.add_argument("--no-save", action="store_true",      help="不保存结果图像")

    # ── info ──
    p_info = sub.add_parser("info", help="显示模型结构与参数统计")
    _add_model_args(p_info)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            lr0=args.lr0,
            lrf=args.lrf,
            optimizer=args.optimizer,
            model=args.model,
            cfg=args.cfg,
            project=args.project,
            name=args.name,
        )

    elif args.mode == "val":
        results = val(
            data=args.data,
            weights=args.weights,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        print(results)

    elif args.mode == "predict":
        predict(
            source=args.source,
            weights=args.weights,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save=not args.no_save,
        )

    elif args.mode == "info":
        show_info(args.model)


if __name__ == "__main__":
    main()
