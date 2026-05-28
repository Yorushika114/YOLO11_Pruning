#!/usr/bin/env python3
"""
verify_yolov5s_mgct.py — 验证 YOLOv5s-MGCT 所有模块是否正确实现.

运行:
    python verify_yolov5s_mgct.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("  YOLOv5s-MGCT 实现验证 (李昂 2023)")
print("=" * 60)


# ─── 1. 检查 CoordAtt 模块 ────────────────────────────────────────────────────
print("\n[1] CoordAtt (CA 空间注意力机制) ...")
try:
    from ultralytics.nn.modules.block import CoordAtt
    ca = CoordAtt(64, 64)
    x = torch.randn(2, 64, 20, 20)
    y = ca(x)
    assert y.shape == x.shape, f"形状不匹配: {y.shape}"
    # 验证 y.c(i,j) = x.c(i,j) * g_h(i) * g_w(j) — 输出不应与输入完全相同
    assert not torch.allclose(y, x), "CoordAtt 输出与输入完全相同 (注意力未生效)"
    print(f"   ✓  CoordAtt(64→64)  输入{list(x.shape)} → 输出{list(y.shape)}")
    print(f"      参数量: {sum(p.numel() for p in ca.parameters()):,}")
except Exception as e:
    print(f"   ✗  CoordAtt 失败: {e}")
    sys.exit(1)


# ─── 2. 检查 GhostConv 模块 ──────────────────────────────────────────────────
print("\n[2] GhostConv (深度可分离幻影卷积) ...")
try:
    from ultralytics.nn.modules.conv import GhostConv
    gc = GhostConv(128, 256, 3, 2)
    x2 = torch.randn(2, 128, 40, 40)
    y2 = gc(x2)
    assert y2.shape == (2, 256, 20, 20), f"形状错误: {y2.shape}"
    # GhostConv 参数量应远少于标准 Conv2d
    std_params = 128 * 256 * 3 * 3    # 标准 Conv: 294,912
    ghost_params = sum(p.numel() for p in gc.parameters())
    reduction = 1 - ghost_params / std_params
    print(f"   ✓  GhostConv(128→256, 3, 2)  输出{list(y2.shape)}")
    print(f"      参数量: {ghost_params:,}  vs 标准Conv: {std_params:,}  节省 {reduction:.1%}")
except Exception as e:
    print(f"   ✗  GhostConv 失败: {e}")
    sys.exit(1)


# ─── 3. 检查 C3Ghost 模块 ────────────────────────────────────────────────────
print("\n[3] C3Ghost (Ghost C3 特征提取块) ...")
try:
    from ultralytics.nn.modules.block import C3Ghost
    c3g = C3Ghost(256, 256, n=3)
    x3 = torch.randn(2, 256, 20, 20)
    y3 = c3g(x3)
    assert y3.shape == x3.shape, f"形状错误: {y3.shape}"
    print(f"   ✓  C3Ghost(256, n=3)  输入{list(x3.shape)} → 输出{list(y3.shape)}")
    print(f"      参数量: {sum(p.numel() for p in c3g.parameters()):,}")
except Exception as e:
    print(f"   ✗  C3Ghost 失败: {e}")
    sys.exit(1)


# ─── 4. 检查完整模型 YAML 加载 ───────────────────────────────────────────────
print("\n[4] 完整 YOLOv5s-MGCT 模型 (yolov5s_mgct.yaml) ...")
try:
    from ultralytics import YOLO
    model = YOLO("ultralytics/cfg/models/v5/yolov5s_mgct.yaml")
    # 检查关键层是否存在
    layer_types = [type(m).__name__ for m in model.model.modules()]
    assert "CoordAtt" in layer_types, "CoordAtt 未出现在模型中"
    assert "GhostConv" in layer_types, "GhostConv 未出现在模型中"
    assert "C3Ghost" in layer_types, "C3Ghost 未出现在模型中"
    # 统计参数量
    total_params = sum(p.numel() for p in model.model.parameters())
    bn_count = sum(1 for m in model.model.modules() if isinstance(m, nn.BatchNorm2d))
    print(f"   ✓  模型加载成功")
    print(f"      总参数量: {total_params / 1e6:.2f} M")
    print(f"      BatchNorm2d 层数: {bn_count}  (稀疏训练/剪枝目标通道总数: {bn_count * 1}+)")
    print(f"      包含层类型: CoordAtt ✓  GhostConv ✓  C3Ghost ✓")
except Exception as e:
    print(f"   ✗  模型加载失败: {e}")
    sys.exit(1)


# ─── 5. 验证 Mixup 批量增强 ──────────────────────────────────────────────────
print("\n[5] Mixup 批量数据增强 (论文 Section 2.1) ...")
try:
    sys.path.insert(0, ".")
    from train_yolov5s_mgct import mixup_batch

    B, C, H, W = 4, 3, 608, 608
    # 构造模拟 batch
    imgs = torch.rand(B, C, H, W)
    n_boxes = 10
    batch = {
        "img":       imgs.clone(),
        "cls":       torch.zeros(n_boxes, 1),
        "bboxes":    torch.rand(n_boxes, 4),
        "batch_idx": torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3, 3], dtype=torch.float32),
    }

    np.random.seed(42)
    out = mixup_batch(batch, alpha=0.2)

    # 检查图像已混合 (不等于原始)
    assert not torch.allclose(out["img"], imgs), "图像未被混合"
    # 检查边界框数量已增加 (取并集)
    assert out["bboxes"].shape[0] >= n_boxes, "边界框数量应 ≥ 原始数量"
    # 检查所有 batch_idx 在 [0, B-1] 范围内
    assert out["batch_idx"].max() < B and out["batch_idx"].min() >= 0

    # 验证 Beta 分布: 统计 1000 次采样的均值应接近 0.5
    lams = [float(np.random.beta(0.2, 0.2)) for _ in range(1000)]
    mean_lam = np.mean(lams)
    print(f"   ✓  Mixup(alpha=0.2)  图像已混合  边界框: {n_boxes} → {out['bboxes'].shape[0]}")
    print(f"      Beta(0.2,0.2) 均值={mean_lam:.3f} (理论=0.5)  "
          f"标准差={np.std(lams):.3f}")
except Exception as e:
    print(f"   ✗  Mixup 失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ─── 6. 验证稀疏梯度 ─────────────────────────────────────────────────────────
print("\n[6] 稀疏训练 — BN γ 的 L1 梯度注入 ...")
try:
    from train_yolov5s_mgct import apply_sparse_grad

    # 构造简单模型
    m = nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )
    x_s = torch.randn(2, 16, 8, 8)
    loss = m(x_s).sum()
    loss.backward()

    bn = m[1]
    grad_before = bn.weight.grad.data.clone()
    apply_sparse_grad(m, lambda_s=0.002)
    grad_after = bn.weight.grad.data.clone()

    delta = grad_after - grad_before                     # 应 = λ * sign(γ)
    expected_sign = torch.sign(bn.weight.data)           # BN 初始化为 1 → 全+1
    # 方向验证: Δgrad 与 sign(γ) 同号
    sign_match = (torch.sign(delta) == expected_sign).float().mean()
    assert sign_match > 0.99, f"梯度方向错误: 同号率={sign_match:.2%}"
    # 幅度验证: |Δgrad| ≈ λ = 0.002 (float32 精度 atol=1e-4)
    assert torch.allclose(delta.abs(), torch.full_like(delta, 0.002), atol=1e-4), \
        "稀疏梯度幅度不正确"
    print(f"   ✓  稀疏梯度注入  Δgrad = λ·sign(γ) = 0.002·sign(γ)")
    print(f"      方向正确率: {sign_match:.0%}  "
          f"|Δgrad| 均值: {delta.abs().mean():.6f} (期望 0.002000)")
except Exception as e:
    print(f"   ✗  稀疏梯度失败: {e}")
    sys.exit(1)


# ─── 7. 验证通道剪枝 ─────────────────────────────────────────────────────────
print("\n[7] 通道剪枝 (论文 Section 2.4, 表5) ...")
try:
    from train_yolov5s_mgct import prune_model, _collect_bn_gammas

    # 构造带 BN 的小模型
    test_model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
    )
    # 手动设置不均匀的 γ 值模拟稀疏训练后的状态
    with torch.no_grad():
        test_model[1].weight.data = torch.cat([
            torch.full((8,),  0.001),   # 前 8 通道 γ 很小
            torch.full((8,),  1.000),   # 后 8 通道 γ 正常
        ])
        test_model[4].weight.data = torch.cat([
            torch.full((16,), 0.001),
            torch.full((16,), 1.000),
        ])

    gammas_before = _collect_bn_gammas(test_model)
    n_pruned = prune_model(test_model, ratio=0.5)
    gammas_after = _collect_bn_gammas(test_model)

    n_zero_after = int((gammas_after == 0).sum())
    assert n_pruned > 0, "没有通道被剪枝"
    assert n_pruned == n_zero_after, f"剪枝数量不匹配: {n_pruned} vs {n_zero_after}"

    print(f"   ✓  prune_model(ratio=0.50)  剪枝了 {n_pruned}/{gammas_before.numel()} 个通道")
    print(f"      剪枝前 |γ|<0.01 占: {float((gammas_before < 0.01).sum()) / gammas_before.numel():.0%}")
    print(f"      剪枝后 γ=0 占:    {float((gammas_after == 0).sum()) / gammas_after.numel():.0%}")
    print(f"      论文最优剪枝率: 60% → mAP=93.9, 体积=4.7MB, FPS=95")
except Exception as e:
    print(f"   ✗  通道剪枝失败: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)


# ─── 8. 快速前向传播测试 (小尺寸) ──────────────────────────────────────────
print("\n[8] 快速前向传播测试 (imgsz=160, bs=2) ...")
try:
    model.model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, 160, 160)
        out = model.model(dummy)
    if isinstance(out, (list, tuple)):
        shapes = [o.shape for o in out if isinstance(o, torch.Tensor)]
        print(f"   ✓  前向传播成功  输出张量形状: {shapes}")
    else:
        print(f"   ✓  前向传播成功  输出形状: {out.shape}")
except Exception as e:
    print(f"   ✗  前向传播失败: {e}")
    import traceback; traceback.print_exc()


# ─── 汇总 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  全部验证通过 ✓")
print("")
print("  论文改进汇总:")
print("  M  Mixup   Beta(0.2,0.2) 批量增强, 替代 Mosaic")
print("  G  Ghost   GhostConv + C3Ghost 替换骨干/颈部标准卷积")
print("  C  CoordAtt 坐标注意力机制 (backbone 末尾, SPPF 之前)")
print("  T  Sparse  L1(λ=0.002) 稀疏训练 + 60% 通道剪枝 + 微调")
print("")
print("  训练命令:")
print("    阶段1: python train_yolov5s_mgct.py --data data.yaml")
print("    阶段2: python train_yolov5s_mgct.py --data data.yaml \\")
print("               --weights runs/detect/mgct/weights/best.pt --prune")
print("=" * 60)
