"""
sdp_yolo.py — SDP-YOLO: Improved Lightweight YOLO11 for Cotton Disease Detection

Paper: "改进的轻量化YOLO11棉花病害检测", Jiang Bibo et al.,
       计算机系统应用, 2026, 35(2):165-174.

Four improvements over YOLO11n:
  §2.1  StarNet backbone       – star-operation lightweight backbone replacing YOLO11n backbone
  §2.2  DRBNCSPELAN4           – dilated reparameterizable neck module replacing C3K2
  §2.3  EPCD detection head    – efficient partial-conv head replacing YOLO11 head
  §2.4  Wise-IoU loss          – dynamic-focusing bbox regression loss replacing CIoU

This file is self-contained: it imports from ultralytics (read-only, no modification)
and registers the custom modules so they can be used for training.

Usage:
  # Training
  python sdp_yolo.py --data cotton.yaml --epochs 300 --batch 16 --device 0

  # Validation
  python sdp_yolo.py --task val --weights runs/sdp_yolo/exp/weights/best.pt --data cotton.yaml

  # Export reparameterized model
  python sdp_yolo.py --task export --weights best.pt --deploy
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# ── Read-only imports from ultralytics (no files are modified) ────────────────
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import SPPF, C2PSA, DFL
from ultralytics.utils.loss import v8DetectionLoss, DFLoss
from ultralytics.utils.tal import (
    TaskAlignedAssigner, dist2bbox, make_anchors, bbox2dist,
)
from ultralytics.models.yolo.detect import DetectionTrainer


# ══════════════════════════════════════════════════════════════════════════════
# §2.1  StarNet Backbone  (Ma et al., "Rewrite the Stars", CVPR 2024)
# ══════════════════════════════════════════════════════════════════════════════

class StarBlock(nn.Module):
    """
    StarNet building block.

    Star operation: (ReLU6(W1 x)) * (W2 x)
    This expands the polynomial order of the mapping efficiently without
    extra depth.  BN follows each depthwise conv; ReLU6 avoids gradient
    explosion (lighter than GELU on edge devices).

    Architecture (Figure 3 in SDP-YOLO paper):
        Input → DWConv(7×7) → BN
              → [FC1(1×1, expand)] → ReLU6 ─┐
              → [FC2(1×1, expand)]           ├─ ⊗ → FC3(1×1, compress)
                                             ┘
              → DWConv(7×7) → BN
              → + Input (residual)
    """

    def __init__(self, c: int, mlp_ratio: int = 3):
        super().__init__()
        mid = int(c * mlp_ratio)
        # spatial mixing
        self.dw1 = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        # star branches (channel expansion then compression)
        self.fc1 = nn.Conv2d(c, mid, 1, bias=False)
        self.fc2 = nn.Conv2d(c, mid, 1, bias=False)
        self.act = nn.ReLU6(inplace=True)
        self.fc3 = nn.Conv2d(mid, c, 1, bias=False)
        # output spatial mixing
        self.dw2 = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn1(self.dw1(x))
        # star operation: act(W1·x) ⊗ (W2·x), per equations (1)–(2) of the paper
        x = self.fc3(self.act(self.fc1(x)) * self.fc2(x))
        x = self.bn2(self.dw2(x))
        return x + residual


class StarStage(nn.Module):
    """
    One StarNet stage: stride-2 Conv (downsampling) followed by a StarBlock.

    Output resolution is halved; channel width changes as specified.
    """

    def __init__(self, c_in: int, c_out: int, mlp_ratio: int = 3):
        super().__init__()
        self.down = Conv(c_in, c_out, 3, 2)   # stride-2 downsampling
        self.block = StarBlock(c_out, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class StarNetBackbone(nn.Module):
    """
    4-stage StarNet backbone for SDP-YOLO (Figure 3 in paper).

    Channel widths: stem=d, stage1=d, stage2=2d, stage3=3d, stage4=4d.
    Outputs P3 (1/8), P4 (1/16), P5 (1/32) for FPN neck.

    With d=32 (nano scale):
        P3 = 64 ch, P4 = 96 ch, P5 = 128 ch
    """

    def __init__(self, c_in: int = 3, d: int = 32, mlp_ratio: int = 3):
        super().__init__()
        self.stem   = Conv(c_in, d, 3, 2)                          # 1/2
        self.stage1 = StarStage(d,     d,     mlp_ratio)            # 1/4
        self.stage2 = StarStage(d,     2 * d, mlp_ratio)            # 1/8  → P3
        self.stage3 = StarStage(2 * d, 3 * d, mlp_ratio)            # 1/16 → P4
        self.stage4 = StarStage(3 * d, 4 * d, mlp_ratio)            # 1/32 → P5
        self.out_channels = [2 * d, 3 * d, 4 * d]                  # c3, c4, c5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x  = self.stem(x)
        x  = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


# ══════════════════════════════════════════════════════════════════════════════
# §2.2  DRBNCSPELAN4  (Neck module replacing C3K2)
#
# Dilated Reparam Block Nested Cross-Stage Progressive Enhanced Layer Aggregation
# Network (DRBNCSPELAN4).  Combines:
#   • UniRepLKNet-style dilated reparameterization for large ERF with low cost
#   • RepNCSPELAN4-style cross-stage partial structure for efficient feature reuse
# ══════════════════════════════════════════════════════════════════════════════

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparameterizable Block (Figure 5 in paper).

    Training mode: 5 parallel dilated branches, all summed.
      Branch configs: (9×9, d=1), (5×5, d=1), (5×5, d=2), (3×3, d=3), (3×3, d=4)
      Equivalent kernel sizes (K_equiv = (k-1)·r + 1): 9, 5, 9, 9, 9

    Inference mode: all branches are re-parameterized into a single 9×9 conv.
      Merge formula (equations 3–5):
        W_final = Σ_branch  pad_to_9x9( sparse_dilated_to_dense(W_branch) ) + BN_fused
    """

    # (kernel_size, dilation_rate) for the 5 training branches
    _BRANCHES: List[Tuple[int, int]] = [(9, 1), (5, 1), (5, 2), (3, 3), (3, 4)]

    def __init__(self, c: int, deploy: bool = False):
        super().__init__()
        self.c = c
        self.deploy = deploy

        if deploy:
            self.repconv = nn.Conv2d(c, c, 9, 1, 4, bias=True)
        else:
            self.convs: nn.ModuleList = nn.ModuleList()
            self.bns:   nn.ModuleList = nn.ModuleList()
            for k, d in self._BRANCHES:
                pad = autopad(k, d=d)
                self.convs.append(nn.Conv2d(c, c, k, 1, pad, dilation=d, bias=False))
                self.bns.append(nn.BatchNorm2d(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.repconv(x)
        return sum(bn(conv(x)) for conv, bn in zip(self.convs, self.bns))

    # ── Re-parameterization helpers ───────────────────────────────────────────

    @staticmethod
    def _fuse_bn_into_conv(
        w: torch.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Absorb BN scale/shift into conv weight and bias."""
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        w_fused = w * scale.reshape(-1, 1, 1, 1)
        b_fused = bn.bias - bn.running_mean * scale
        return w_fused, b_fused

    @staticmethod
    def _dilated_to_9x9(w: torch.Tensor, k: int, d: int) -> torch.Tensor:
        """
        Convert a (C,C,k,k) dilated-conv kernel (dilation=d) to its
        equivalent (C,C,9,9) dense kernel.  Equation (3): K_equiv = (k-1)·r + 1.
        Non-kernel positions are zero.
        """
        c_out, c_in = w.shape[:2]
        k_eq  = (k - 1) * d + 1               # effective kernel size
        pad   = (9 - k_eq) // 2               # zero-padding to reach 9×9
        w_eq  = torch.zeros(c_out, c_in, 9, 9, device=w.device, dtype=w.dtype)
        for i in range(k):
            for j in range(k):
                w_eq[:, :, i * d + pad, j * d + pad] = w[:, :, i, j]
        return w_eq

    def reparameterize(self) -> None:
        """
        Merge all training branches into a single 9×9 conv (call before deployment).
        Sets self.deploy = True and frees the training branches from memory.
        """
        if self.deploy:
            return

        merged_w = torch.zeros(
            self.c, self.c, 9, 9,
            device=self.convs[0].weight.device,
            dtype=self.convs[0].weight.dtype,
        )
        merged_b = torch.zeros(self.c, device=self.convs[0].weight.device)

        for (k, d), conv, bn in zip(self._BRANCHES, self.convs, self.bns):
            w_fused, b_fused = self._fuse_bn_into_conv(conv.weight, bn)
            merged_w += self._dilated_to_9x9(w_fused, k, d)
            merged_b += b_fused

        self.repconv = nn.Conv2d(self.c, self.c, 9, 1, 4, bias=True)
        self.repconv.weight.data = merged_w
        self.repconv.bias.data   = merged_b

        self.deploy = True
        del self.convs, self.bns


class DRBNBottleneck(nn.Module):
    """
    Dilated-Reparam Bottleneck (right diagram in Figure 4).

    DilatedReparamBlock → Conv(3×3) → (+) residual
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, deploy: bool = False):
        super().__init__()
        self.drb = DilatedReparamBlock(c1, deploy=deploy)
        self.cv  = Conv(c1, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv(self.drb(x))
        return x + y if self.add else y


class DRBNCSP(nn.Module):
    """
    Cross-Stage-Partial wrapper around DRBNBottleneck (middle diagram in Figure 4).

    Splits input → two branches:
      Branch 1: Conv(1×1) → channel identity
      Branch 2: Conv(1×1) → n × DRBNBottleneck
    Concat → Conv(1×1) → output
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, deploy: bool = False):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m   = nn.Sequential(*[DRBNBottleneck(c_, c_, shortcut, deploy) for _ in range(n)])
        self.cv3 = Conv(c_ + c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.cv1(x), self.m(self.cv2(x))], 1))


class DRBNCSPELAN4(nn.Module):
    """
    DRBNCSPELAN4 — Dilated Reparam Block Nested Cross-Stage Progressive
    Enhanced Layer Aggregation Network (left diagram in Figure 4).

    Mirrors the RepNCSPELAN4 architecture (YOLOv9) but substitutes each
    RepCSP sub-branch with DRBNCSP (dilated-reparam version).

    Args:
        c1  (int): Input channels.
        c2  (int): Output channels.
        c3  (int): Intermediate split channels (must be even).
        c4  (int): Inner CSP output channels.
        n   (int): Repeat count for DRBNBottleneck inside DRBNCSP.
        deploy (bool): If True, re-parameterized (inference) mode.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        c3: int,
        c4: int,
        n: int = 1,
        deploy: bool = False,
    ):
        super().__init__()
        self.c   = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(DRBNCSP(c3 // 2, c4, n, deploy=deploy), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(DRBNCSP(c4,      c4, n, deploy=deploy), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))          # split → [c3//2, c3//2]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def reparameterize(self) -> None:
        """Recursively reparameterize all DilatedReparamBlocks."""
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.reparameterize()


# ══════════════════════════════════════════════════════════════════════════════
# §2.3  EPCD Detection Head  (Efficient Partial Convolution Detection)
# ══════════════════════════════════════════════════════════════════════════════

class PConv(nn.Module):
    """
    Partial Convolution (Chen et al., "Run, Don't Walk", CVPR 2023).

    Only the first 1/4 of input channels undergo a standard conv;
    the remaining 3/4 pass through unchanged (identity).
    After the partial conv, channels are concatenated back.

    Computation cost ≈ 1/4 of a full conv of the same size.
    Structure shown in Figure 8 of the paper.
    """

    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.c_p  = max(1, c // 4)                             # partial channels
        self.conv = nn.Conv2d(self.c_p, self.c_p, k, 1, k // 2, bias=False)
        self.bn   = nn.BatchNorm2d(self.c_p)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :self.c_p]
        x2 = x[:, self.c_p:]
        x1 = self.act(self.bn(self.conv(x1)))
        return torch.cat([x1, x2], dim=1)


class EPCDHead(nn.Module):
    """
    Efficient Partial Convolution Detection head (Figure 7 in paper).

    Replaces YOLO11's two-branch head (DWConv paths for bbox and cls) with a
    single unified path: PConv(3×3) → Conv(1×1) → all predictions.

    This halves the detection-head computation and reduces parameters,
    as described in §2.3 of the paper.

    The output format is identical to the standard Detect head, ensuring
    full compatibility with the ultralytics loss and validation pipeline.
    """

    dynamic = False
    export   = False
    format   = None
    shape    = None
    anchors  = torch.empty(0)
    strides  = torch.empty(0)

    def __init__(self, nc: int = 6, ch: Tuple[int, ...] = ()):
        super().__init__()
        self.nc      = nc
        self.nl      = len(ch)
        self.reg_max = 16
        self.no      = nc + self.reg_max * 4
        self.stride  = torch.zeros(self.nl)
        self.legacy  = False

        # One path per scale: PConv → 1×1 Conv → (4*reg_max + nc) outputs
        self.cv = nn.ModuleList(
            nn.Sequential(PConv(x, k=3), nn.Conv2d(x, self.no, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: List[torch.Tensor]) -> Union[List[torch.Tensor], Tuple]:
        for i in range(self.nl):
            x[i] = self.cv[i](x[i])   # shape: (B, no, Hi, Wi)

        if self.training:
            return x                   # raw feature maps for loss computation

        return self._inference(x)

    def _inference(self, x: List[torch.Tensor]) -> Union[torch.Tensor, Tuple]:
        shape = x[0].shape             # (B, no, H0, W0)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return dist2bbox(bboxes, anchors, xywh=not self.export, dim=1)

    def bias_init(self) -> None:
        """Initialize detection biases (requires stride to be set first)."""
        for seq, s in zip(self.cv, self.stride):
            last = seq[-1]              # the nn.Conv2d(x, self.no, 1)
            last.bias.data[:self.reg_max * 4] += math.log(8 / (640 / float(s)) ** 2)
            last.bias.data[self.reg_max * 4:] += math.log(0.6 / (self.nc - 0.99999))


# ══════════════════════════════════════════════════════════════════════════════
# §2.4  Wise-IoU Loss  (Tong et al. 2023, adapted in SDP-YOLO §2.4)
# ══════════════════════════════════════════════════════════════════════════════

def wise_iou(
    pred_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    alpha: float = 1.9,
    delta: float = 3.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute element-wise Wise-IoU focal loss (equations 9–11 in paper).

    Args:
        pred_xyxy   : (N, 4) predicted boxes in xyxy format.
        target_xyxy : (N, 4) ground-truth boxes in xyxy format.
        alpha       : hyper-parameter α > 1 (default 1.9).
        delta       : hyper-parameter δ > α (default 3.0).
        eps         : numerical stability constant.

    Returns:
        (N,) per-sample Wise-IoU focal losses (not 1 - IoU).

    Mathematical formulation:
        R_WIoU = (1 - IoU) · exp( dist² / (Wg² + Hg²) )          (eq. 11)
        β_i    = mean(1 - IoU) / (1 - IoU_i)   [L*_IoU / L_IoU]  (eq. 10)
        r_i    = β_i / (δ · α^(β_i − δ))                          (eq. 9)
        L_WIoU = r.detach() · R_WIoU                              (final)
    """
    # ── Intersection ─────────────────────────────────────────────────────────
    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # ── Box dimensions ───────────────────────────────────────────────────────
    pw  = (pred_xyxy[:, 2]   - pred_xyxy[:, 0]).clamp(eps)
    ph  = (pred_xyxy[:, 3]   - pred_xyxy[:, 1]).clamp(eps)
    gw  = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(eps)
    gh  = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(eps)
    union = pw * ph + gw * gh - inter + eps
    iou   = inter / union                                   # (N,)

    # ── Center distance ──────────────────────────────────────────────────────
    px    = (pred_xyxy[:, 0]   + pred_xyxy[:, 2])   / 2
    py    = (pred_xyxy[:, 1]   + pred_xyxy[:, 3])   / 2
    gx    = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
    gy    = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2
    dist2 = (px - gx) ** 2 + (py - gy) ** 2

    # ── R_WIoU  (geometric focal term, equation 11) ──────────────────────────
    r_wiou = (1.0 - iou) * torch.exp(dist2 / (gw ** 2 + gh ** 2 + eps))

    # ── Dynamic focusing weight r  (equations 9–10) ──────────────────────────
    # β = L*_IoU / L_IoU  where L*_IoU = batch-mean IoU loss
    l_iou  = (1.0 - iou).detach()                           # no gradient
    l_star = l_iou.mean()                                   # L*_IoU
    beta   = l_star / (l_iou + eps)                         # (N,) ≥ 0
    # r(β): peaks near β = δ; suppresses outliers (β→0) and easy samples (β→∞)
    r = (beta / (delta * alpha ** (beta - delta))).clamp(1e-6, 10.0).detach()

    return r * r_wiou


class WiseIoUBboxLoss(nn.Module):
    """
    Bounding-box loss: Wise-IoU + Distribution Focal Loss (DFL).

    Drop-in replacement for ultralytics BboxLoss; replaces CIoU with Wise-IoU.
    """

    def __init__(self, reg_max: int = 16, alpha: float = 1.9, delta: float = 3.0):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.alpha    = alpha
        self.delta    = delta

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # Wise-IoU replaces CIoU
        wiou = wise_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask],
            alpha=self.alpha, delta=self.delta,
        )
        loss_iou = (wiou.unsqueeze(-1) * weight).sum() / target_scores_sum

        # DFL loss (unchanged from standard pipeline)
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                ) * weight
            ).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class SDPYOLOLoss(v8DetectionLoss):
    """
    Detection loss for SDP-YOLO: standard cls + DFL losses, with Wise-IoU
    replacing CIoU for bounding-box regression.

    Subclasses v8DetectionLoss and only overrides the box-loss branch.
    The full __call__ is re-implemented to avoid the CIoU+PIoU mixture
    present in the modified ultralytics loss.py.
    """

    def __init__(self, model: nn.Module):
        super().__init__(model)
        # Replace the bbox_loss with Wise-IoU version
        self.wise_bbox_loss = WiseIoUBboxLoss(self.reg_max).to(self.device)

    def __call__(
        self,
        preds: Any,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute total loss (box + cls + dfl) using Wise-IoU."""
        loss       = torch.zeros(3, device=self.device)
        feats      = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype      = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)   # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predictions
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)   # xyxy

        # Task-aligned assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss (BCE)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Wise-IoU bounding-box regression loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.wise_bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * batch_size, loss.detach()


# ══════════════════════════════════════════════════════════════════════════════
# §3  Complete SDP-YOLO Model
# ══════════════════════════════════════════════════════════════════════════════

class SDPYOLOModel(nn.Module):
    """
    SDP-YOLO full detection model (Figure 2 in paper).

    Architecture:
        Backbone : StarNet (4 stages, star-operation blocks)
        Neck     : PAFPN with DRBNCSPELAN4 (top-down + bottom-up)
        Head     : EPCDHead at P3/P4/P5 scales

    The forward method follows the ultralytics BaseModel convention:
        forward(dict)   →  (loss_sum, loss_items)   [training]
        forward(tensor) →  predictions              [inference]

    The model[-1] accessor returns the EPCDHead for loss function setup.
    """

    def __init__(self, nc: int = 6, d: int = 32, deploy: bool = False):
        super().__init__()
        self.nc     = nc
        self.deploy = deploy

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = StarNetBackbone(3, d=d)
        c3, c4, c5   = self.backbone.out_channels   # e.g. 64, 96, 128 for d=32

        # ── Backbone tail (kept from YOLO11n) ─────────────────────────────────
        self.sppf  = SPPF(c5, c5, k=5)
        self.c2psa = C2PSA(c5, c5)

        # ── Neck: top-down path ───────────────────────────────────────────────
        # Upsample P5 and fuse with P4
        self.up1  = nn.Upsample(scale_factor=2, mode="nearest")
        self.drb1 = DRBNCSPELAN4(c5 + c4, c4, c4, c4 // 2, n=1, deploy=deploy)
        # Upsample fused P4 and fuse with P3
        self.up2  = nn.Upsample(scale_factor=2, mode="nearest")
        self.drb2 = DRBNCSPELAN4(c4 + c3, c3, c3, c3 // 2, n=1, deploy=deploy)

        # ── Neck: bottom-up path ─────────────────────────────────────────────
        # Downsample P3-out and fuse with P4-neck
        self.dn1  = Conv(c3, c3, 3, 2)
        self.drb3 = DRBNCSPELAN4(c3 + c4, c4, c4, c4 // 2, n=1, deploy=deploy)
        # Downsample P4-out and fuse with P5-enhanced
        self.dn2  = Conv(c4, c4, 3, 2)
        self.drb4 = DRBNCSPELAN4(c4 + c5, c5, c5, c5 // 2, n=1, deploy=deploy)

        # ── Detection head (3 scales) ─────────────────────────────────────────
        self._head = EPCDHead(nc=nc, ch=(c3, c4, c5))

        # model[-1] accessor required by SDPYOLOLoss / v8DetectionLoss
        # Using a plain list avoids double-counting parameters.
        self.model = [self._head]

        # Required trainer attributes (set lazily)
        self.names  = {i: str(i) for i in range(nc)}
        self.args   = None
        self.inplace = True
        self.task   = "detect"

        # AutoBackend compat: loaded .pt is inspected for model.yaml / model.stride
        self.yaml   = {"channels": 3, "nc": nc, "task": "detect"}

        # Initialize strides and detection biases
        self._init_strides()
        self.stride = self._head.stride

        # Lazy criterion
        self.criterion = None

    # ── Stride initialization ─────────────────────────────────────────────────

    def _init_strides(self) -> None:
        """
        Run one forward pass on a dummy image to measure per-level strides,
        then initialise detection biases.
        """
        device = next(self.parameters()).device
        dummy  = torch.zeros(1, 3, 256, 256, device=device)
        with torch.no_grad():
            feats = self._neck_forward(dummy)
        strides = torch.tensor(
            [256 / f.shape[-2] for f in feats], dtype=torch.float32, device=device
        )
        self._head.stride = strides
        self._head.bias_init()

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _neck_forward(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """Backbone + neck forward; returns [P3_out, P4_out, P5_out]."""
        p3, p4, p5 = self.backbone(x)
        p5e = self.c2psa(self.sppf(p5))          # P5 enhanced

        # top-down
        n1  = self.drb1(torch.cat([self.up1(p5e), p4], 1))   # fused P4
        n2  = self.drb2(torch.cat([self.up2(n1),  p3], 1))   # fused P3 → small

        # bottom-up
        n3  = self.drb3(torch.cat([self.dn1(n2), n1],  1))   # fused P4 → medium
        n4  = self.drb4(torch.cat([self.dn2(n3), p5e], 1))   # fused P5 → large

        return [n2, n3, n4]

    # ── Main forward ─────────────────────────────────────────────────────────

    def forward(
        self, x: Union[torch.Tensor, Dict], *args, **kwargs
    ) -> Union[torch.Tensor, Tuple, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Dispatch: dict input → training loss; tensor input → predictions.
        Mirrors ultralytics BaseModel.forward() convention.
        """
        if isinstance(x, dict):
            return self.loss(x)
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple]:
        feats = self._neck_forward(x)
        return self._head(feats)

    def loss(
        self, batch: Dict[str, torch.Tensor], preds: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.criterion is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.predict(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self) -> SDPYOLOLoss:
        return SDPYOLOLoss(self)

    # ── Deployment ────────────────────────────────────────────────────────────

    def reparameterize(self) -> "SDPYOLOModel":
        """
        Merge all DilatedReparamBlocks into single 9×9 convs.
        Call once after training, before deployment.
        """
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.reparameterize()
        self.deploy = True
        return self

    # ── Convenience ──────────────────────────────────────────────────────────

    def info(self, verbose: bool = True) -> None:
        """Print parameter count and layer summary."""
        n_params = sum(p.numel() for p in self.parameters())
        n_params_m = n_params / 1e6
        if verbose:
            print(f"SDP-YOLO  nc={self.nc}  d={self.backbone.out_channels[0] // 2}")
            print(f"  Backbone  : StarNet")
            print(f"  Neck      : DRBNCSPELAN4 × 4")
            print(f"  Head      : EPCD (3 scales)")
            print(f"  Parameters: {n_params_m:.2f}M")
            print(f"  Strides   : {self._head.stride.tolist()}")

    def is_fused(self, thresh: int = 10) -> bool:
        """Return True when fewer than thresh BatchNorm2d layers remain (i.e. already fused)."""
        return sum(isinstance(m, nn.BatchNorm2d) for m in self.modules()) < thresh

    def fuse(self, verbose: bool = True) -> "SDPYOLOModel":
        """Fuse Conv+BN layers for faster inference (mirrors BaseModel.fuse)."""
        from ultralytics.utils.torch_utils import fuse_conv_and_bn
        if not self.is_fused():
            for m in self.modules():
                if isinstance(m, Conv) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse
        if verbose:
            self.info()
        return self


# ══════════════════════════════════════════════════════════════════════════════
# §4  Training Integration  (DetectionTrainer subclass)
# ══════════════════════════════════════════════════════════════════════════════

class SDPYOLOTrainer(DetectionTrainer):
    """
    Minimal DetectionTrainer subclass for SDP-YOLO.

    Overrides only get_model() to return SDPYOLOModel instead of the
    standard YAML-based DetectionModel.  All other training logic
    (dataloaders, optimizer, LR scheduler, validation, checkpointing)
    is inherited unchanged from DetectionTrainer / BaseTrainer.
    """

    def get_model(
        self,
        cfg: Optional[str] = None,
        weights: Optional[str] = None,
        verbose: bool = True,
    ) -> SDPYOLOModel:
        nc = self.data["nc"]
        # _d is set by the caller (train/val functions) as an instance attribute.
        # Falls back to 32 (nano scale) if not provided.
        d = getattr(self, "_d", 32)
        # _weights may also be set by the caller for checkpoint resuming.
        w = weights or getattr(self, "_weights", None)

        model = SDPYOLOModel(nc=nc, d=d)

        if w:
            ckpt = torch.load(w, map_location="cpu")
            state = ckpt.get("model", ckpt)
            if hasattr(state, "state_dict"):
                state = state.state_dict()
            missing, unexpected = model.load_state_dict(state, strict=False)
            if verbose and (missing or unexpected):
                print(f"  Loaded weights from {w}")
                if missing:
                    print(f"  Missing keys  : {missing[:5]} ...")
                if unexpected:
                    print(f"  Unexpected keys: {unexpected[:5]} ...")

        if verbose:
            model.info()
        return model


# ══════════════════════════════════════════════════════════════════════════════
# §5  Command-line Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SDP-YOLO: Improved Lightweight YOLO11 for Cotton Disease Detection"
    )
    p.add_argument("--task",    default="train",  choices=["train", "val", "export"],
                   help="Task to perform")
    p.add_argument("--data",    default="/gemini/code/Torch-Pruning-master/examples/yolov8/data.yaml",
                   help="Path to dataset YAML (e.g. cotton.yaml)")
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--imgsz",   type=int,   default=160)
    p.add_argument("--device",  default="0",
                   help="CUDA device (e.g. '0', '0,1', 'cpu')")
    p.add_argument("--workers", type=int,   default=8)
    p.add_argument("--project", default="runs/sdp_yolo",
                   help="Save directory root")
    p.add_argument("--name",    default="exp",
                   help="Experiment name")
    p.add_argument("--weights", default=None,
                   help="Path to pretrained or checkpoint weights (.pt)")
    p.add_argument("--d",       type=int,   default=32,
                   help="StarNet base channel width (nano=32, small=48, medium=64)")
    p.add_argument("--deploy",  action="store_true",
                   help="Reparameterize DilatedReparamBlocks after training")
    p.add_argument("--lr0",     type=float, default=0.01,
                   help="Initial learning rate")
    p.add_argument("--momentum",type=float, default=0.937)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--no-augment",   action="store_true",
                   help="Disable mosaic augmentation")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    """Launch training via SDPYOLOTrainer."""
    overrides = {
        # "model" must be a non-None string; BaseTrainer.__init__ calls
        # check_model_file_from_stem(self.args.model) and Path(None) raises TypeError.
        # "sdp_yolo" is not in GITHUB_ASSETS_STEMS so it passes through unchanged;
        # setup_model() then calls get_model(cfg="sdp_yolo") which our override handles.
        "model":          "sdp_yolo",
        "data":           args.data,
        "epochs":         args.epochs,
        "batch":          args.batch,
        "imgsz":          args.imgsz,
        "device":         args.device,
        "workers":        args.workers,
        "project":        args.project,
        "name":           args.name,
        "lr0":            args.lr0,
        "momentum":       args.momentum,
        "weight_decay":   args.weight_decay,
        "optimizer":      "SGD",
        "warmup_momentum": 0.8,
        "mosaic":         0.0 if args.no_augment else 1.0,
    }
    trainer = SDPYOLOTrainer(overrides=overrides)
    trainer._d = args.d           # pass StarNet base channel width to get_model()
    trainer._weights = args.weights
    trainer.train()


def val(args: argparse.Namespace) -> None:
    """
    Run validation on a saved SDP-YOLO checkpoint.

    Builds the model from the checkpoint, then runs the full validation loop
    by leveraging the trainer's built-in infrastructure.
    """
    if not args.weights:
        raise ValueError("--weights is required for the val task")

    overrides = {
        "model":   "sdp_yolo",   # non-None placeholder; same reason as in train()
        "data":    args.data,
        "imgsz":   args.imgsz,
        "batch":   args.batch,
        "device":  args.device,
        "workers": args.workers,
        "project": args.project,
        "name":    args.name + "_val",
        "epochs":  1,
    }
    trainer = SDPYOLOTrainer(overrides=overrides)
    trainer._d       = args.d
    trainer._weights = args.weights

    # Initialize trainer internals (loads data config, sets device, etc.)
    trainer._setup_train(world_size=1)

    # Override the model with checkpoint weights
    ckpt   = torch.load(args.weights, map_location="cpu")
    state  = ckpt.get("model", ckpt)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    trainer.model.load_state_dict(state, strict=False)
    trainer.model.eval()

    metrics, fitness = trainer.validate()
    print(f"\nValidation results  (fitness={fitness:.4f}):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def export_model(args: argparse.Namespace) -> None:
    """Export a reparameterized SDP-YOLO model to torchscript / onnx."""
    if not args.weights:
        raise ValueError("--weights required for export task")

    ckpt  = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("model", ckpt)
    nc    = ckpt.get("nc", 6)
    d     = ckpt.get("d",  args.d)

    model = SDPYOLOModel(nc=nc, d=d)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)

    if args.deploy:
        print("Reparameterizing DilatedReparamBlocks ...")
        model.reparameterize()

    model.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)

    # TorchScript
    ts_path = Path(args.weights).with_suffix(".torchscript.pt")
    scripted = torch.jit.trace(model, dummy)
    scripted.save(str(ts_path))
    print(f"Saved TorchScript model → {ts_path}")

    # ONNX
    try:
        onnx_path = Path(args.weights).with_suffix(".onnx")
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
            opset_version=12,
        )
        print(f"Saved ONNX model → {onnx_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def main() -> None:
    args = _parse_args()
    if args.task == "train":
        train(args)
    elif args.task == "val":
        val(args)
    elif args.task == "export":
        export_model(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
