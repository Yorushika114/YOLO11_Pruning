"""
YOLOv8-Corn: 基于剪枝与蒸馏的轻量化玉米病害检测方法
论文: 李龙海等, 农业工程学报, 2025, 41(17): 194-202.
doi: 10.11975/j.issn.1002-6819.202504008

【完整流程(四阶段)】
─────────────────────────────────────────────────────────
Step 0  基础训练       (可选, base_train_epochs > 0 时启用)
Step 1  稀疏化训练     (L1 正则使 BN gamma 稀疏)
Step 2  通道剪枝       (按 |gamma| 阈值剪通道)
Step 3  回调微调       (恢复剪枝损失的精度)
Step 4  CWD 蒸馏微调   (可选, do_distill=True 时启用)

【自定义数据集快速配置】
─────────────────────────────────────────────────────────
方式 A: 直接指定已有的 data.yaml 路径
    prune_yolov8(model_path='yolov8n.pt', data_yaml='C:/mydata/data.yaml',
                 base_train_epochs=200)

方式 B: 从数据集文件夹自动生成 data.yaml
    from yolov8_corn import DatasetConfig
    cfg = DatasetConfig(
        dataset_dir = 'C:/mydata',
        class_names = ['leaf_spot', 'gray_spot', 'mosaic', 'rust'],
    )
    cfg.check()
    data_yaml = cfg.generate_yaml()
    prune_yolov8(model_path='yolov8n.pt', data_yaml=data_yaml,
                 base_train_epochs=200)

方式 C: 修复已有 data.yaml 里的路径
    from yolov8_corn import DatasetConfig
    data_yaml = DatasetConfig.fix_yaml_path('C:/mydata/data.yaml')

【数据集目录结构】
─────────────────────────────────────────────────────────
dataset_dir/
  ├── images/
  │   ├── train/   ← 训练图片 (.jpg / .png)
  │   ├── val/     ← 验证图片
  │   └── test/    (可选)
  └── labels/
      ├── train/   ← YOLO 格式标注 (.txt)
      ├── val/
      └── test/    (可选)

标注格式 (每行): <class_id> <cx> <cy> <w> <h>  (归一化 0~1)
─────────────────────────────────────────────────────────
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 第零部分: 数据集配置工具
# ============================================================

class DatasetConfig:
    """
    自定义数据集配置工具，用于生成或修复 YOLO 格式的 data.yaml。

    用法:
        cfg = DatasetConfig('D:/mydata', ['leaf_spot', 'rust'])
        cfg.check()
        yaml_path = cfg.generate_yaml()
    """

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

    def __init__(
        self,
        dataset_dir: str,
        class_names: List[str],
        train_dir: str = 'images/train',
        val_dir:   str = 'images/val',
        test_dir:  Optional[str] = None,
    ):
        """
        Args:
            dataset_dir  : 数据集根目录（绝对路径或相对路径均可）
            class_names  : 类别名称列表，顺序与标注文件中的 class_id 对应
            train_dir    : 相对于 dataset_dir 的训练集图片目录
            val_dir      : 相对于 dataset_dir 的验证集图片目录
            test_dir     : 相对于 dataset_dir 的测试集图片目录（可选）
        """
        self.dataset_dir = Path(dataset_dir).resolve()
        self.class_names = class_names
        self.train_dir   = train_dir
        self.val_dir     = val_dir
        self.test_dir    = test_dir

    # --------------------------------------------------------
    # check(): 检查目录结构与图片/标注对应情况
    # --------------------------------------------------------

    def check(self) -> bool:
        """
        检查数据集目录结构是否符合 YOLO 要求。
        打印各子集的图片数量、标注数量及缺失情况。

        Returns:
            True 表示结构正常，False 表示存在问题
        """
        print(f"\n[DatasetConfig] 检查数据集: {self.dataset_dir}")
        ok = True

        subsets = [('train', self.train_dir), ('val', self.val_dir)]
        if self.test_dir:
            subsets.append(('test', self.test_dir))

        for split, img_rel in subsets:
            img_dir = self.dataset_dir / img_rel
            lbl_dir = self.dataset_dir / img_rel.replace('images', 'labels')

            if not img_dir.exists():
                print(f"  [✗] {split} 图片目录不存在: {img_dir}")
                ok = False
                continue

            imgs   = [f for f in img_dir.iterdir()
                      if f.suffix.lower() in self.IMG_EXTS]
            labels = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []

            # 找出没有对应标注的图片
            img_stems = {f.stem for f in imgs}
            lbl_stems = {f.stem for f in labels}
            missing   = img_stems - lbl_stems

            status = '✓' if not missing else '!'
            print(f"  [{status}] {split:<6s}: "
                  f"{len(imgs)} 张图片, {len(labels)} 个标注"
                  + (f", 缺少 {len(missing)} 个标注" if missing else ""))

            if missing and len(missing) <= 5:
                for s in list(missing)[:5]:
                    print(f"         缺失: {s}")

        print(f"  类别 ({len(self.class_names)}): {self.class_names}")
        return ok

    # --------------------------------------------------------
    # generate_yaml(): 生成 data.yaml
    # --------------------------------------------------------

    def generate_yaml(self, output_path: Optional[str] = None) -> str:
        """
        在数据集根目录（或指定路径）生成 data.yaml。

        Args:
            output_path: yaml 保存路径，默认为 dataset_dir/data.yaml

        Returns:
            生成的 yaml 文件的绝对路径字符串
        """
        if output_path is None:
            output_path = self.dataset_dir / 'data.yaml'
        output_path = Path(output_path)

        lines = [
            f"path: {self.dataset_dir.as_posix()}",
            f"train: {self.train_dir}",
            f"val: {self.val_dir}",
        ]
        if self.test_dir:
            lines.append(f"test: {self.test_dir}")

        lines += [
            "",
            f"nc: {len(self.class_names)}",
            "names:",
        ]
        for name in self.class_names:
            lines.append(f"  - {name}")

        output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f"[DatasetConfig] data.yaml 已生成: {output_path}")
        return str(output_path)

    # --------------------------------------------------------
    # fix_yaml_path(): 修复已有 yaml 里的 path 字段
    # --------------------------------------------------------

    @staticmethod
    def fix_yaml_path(yaml_path: str, new_dataset_dir: Optional[str] = None) -> str:
        """
        修复已有 data.yaml 中的 path 字段（当数据集移动后路径失效时使用）。

        Args:
            yaml_path       : 现有 data.yaml 的路径
            new_dataset_dir : 新的数据集根目录，默认使用 yaml 文件所在目录

        Returns:
            修复后的 yaml 文件路径（原地修改并返回）
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"找不到 yaml 文件: {yaml_path}")

        if new_dataset_dir is None:
            new_dataset_dir = yaml_path.parent.resolve()
        else:
            new_dataset_dir = Path(new_dataset_dir).resolve()

        lines = yaml_path.read_text(encoding='utf-8').splitlines()
        new_lines = []
        fixed = False
        for line in lines:
            if line.startswith('path:'):
                new_lines.append(f"path: {new_dataset_dir.as_posix()}")
                fixed = True
            else:
                new_lines.append(line)

        if not fixed:
            new_lines.insert(0, f"path: {new_dataset_dir.as_posix()}")

        yaml_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
        print(f"[DatasetConfig] 已更新 path → {new_dataset_dir}")
        return str(yaml_path)


# ============================================================
# 第一部分: 论文算法模块
# ============================================================

# ------------------------------------------------------------
# 1. DCNv2 - 可变形卷积 v2
#    论文公式 (1): y(p) = Σ_n w_n · x(p + p_n + Δp_n) · Δm_n
# ------------------------------------------------------------

class DCNv2(nn.Module):
    """可变形卷积 v2，通过可学习偏移 Δp_n 和调制掩码 Δm_n 自适应对齐特征。"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, groups: int = 1):
        super().__init__()
        self.stride  = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride, padding, groups=groups, bias=False)
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                      kernel_size, stride, padding, bias=True)
        self.mask_conv   = nn.Conv2d(in_channels, kernel_size * kernel_size,
                                      kernel_size, stride, padding, bias=True)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask   = torch.sigmoid(self.mask_conv(x))
        try:
            from torchvision.ops import deform_conv2d
            return deform_conv2d(x, offset, self.conv.weight, self.conv.bias,
                                  stride=self.stride, padding=self.padding, mask=mask)
        except ImportError:
            return self.conv(x)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 3, 1)
        self.cv2 = ConvBNAct(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# ------------------------------------------------------------
# 2. C2f-DCNv2
# ------------------------------------------------------------

class DCNBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__()
        self.cv1 = ConvBNAct(c1, c2, 1, 1)
        self.dcn = DCNv2(c2, c2, 3, 1, 1)
        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.act(self.bn(self.dcn(self.cv1(x))))
        return x + out if self.add else out


class C2fDCNv2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c   = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(DCNBottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------------------------
# 3. DySample - 动态上采样
#    论文公式 (2)(3): σ = linear(x), S = G + σ
# ------------------------------------------------------------

class DySample(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int = 2, groups: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        self.groups = groups
        self.offset = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, groups * 2 * scale_factor * scale_factor, 1)
        )
        for m in self.offset.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        Ho, Wo = H * self.scale_factor, W * self.scale_factor
        sigma = self.offset(x)
        sigma = sigma.view(B, self.groups, 2,
                           self.scale_factor, self.scale_factor, H, W)
        # permute → (B, groups, H, sf_h, W, sf_w, 2), average over groups → (B, H, sf_h, W, sf_w, 2)
        sigma = sigma.permute(0, 1, 5, 3, 6, 4, 2).mean(dim=1).reshape(B, Ho, Wo, 2).tanh()
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, Ho, device=x.device),
            torch.linspace(-1, 1, Wo, device=x.device),
            indexing='ij')
        G = torch.stack([gx, gy], -1).unsqueeze(0).expand(B, -1, -1, -1)
        S = (G + sigma).clamp(-1, 1)
        x_up = F.interpolate(x, size=(Ho, Wo), mode='bilinear', align_corners=True)
        return F.grid_sample(x_up, S, mode='bilinear', align_corners=True,
                              padding_mode='border')


# ------------------------------------------------------------
# 4. CWDLoss - 通道知识蒸馏损失
#    论文公式 (4)(5)
# ------------------------------------------------------------

class CWDLoss(nn.Module):
    """通道知识蒸馏: 各通道 softmax 后计算 KL 散度。"""

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        assert feat_s.shape == feat_t.shape, \
            f"student {feat_s.shape} != teacher {feat_t.shape}"
        B, C, H, W = feat_s.shape
        p_t = F.softmax(feat_t.view(B, C, -1) / self.T, dim=-1)
        p_s = F.softmax(feat_s.view(B, C, -1) / self.T, dim=-1)
        kl  = p_t * (torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8))
        return (self.T ** 2 / C) * kl.sum(-1).mean()


# ------------------------------------------------------------
# 5. MGDLoss - 掩码生成蒸馏损失
#    论文公式 (6)
# ------------------------------------------------------------

class MGDLoss(nn.Module):
    """掩码生成蒸馏: 学生从遮盖特征重建教师特征。"""

    def __init__(self, student_channels: int, teacher_channels: int,
                 mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.align     = nn.Conv2d(student_channels, teacher_channels, 1)
        self.generator = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, 1),
        )
        nn.init.kaiming_normal_(self.align.weight)
        for m in self.generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        fa = self.align(feat_s)
        M  = (torch.rand(fa.shape[0], 1, fa.shape[2], fa.shape[3],
                          device=fa.device) > self.mask_ratio).float()
        return F.mse_loss(self.generator(fa * M), feat_t.detach())


# ============================================================
# 第二部分: Ultralytics 集成剪枝工具
# ============================================================

def make_sparse_trainer(sparsity_rate: float = 0.02):
    """
    返回注入 L1 稀疏正则的 DetectionTrainer 子类。

    实现原理:
        L1 正则 sr·Σ|γ| 对 γ 的梯度 = sr·sign(γ)。
        在 optimizer_step 执行梯度裁剪前，将 sr·sign(γ) 加到 BN gamma 的梯度上，
        等价于在 loss 中加 L1 项，但无需修改前向/反向传播。
    """
    try:
        from ultralytics.models.yolo.detect.train import DetectionTrainer
    except ImportError:
        raise ImportError("未找到 ultralytics，请在项目目录下运行或执行 pip install ultralytics")

    class SparseTrainer(DetectionTrainer):
        _sparsity_rate = sparsity_rate  # 类变量，避免与 Ultralytics 参数体系冲突

        def optimizer_step(self):
            # 在梯度裁剪前注入 sr·sign(γ)，实现 BN gamma 稀疏化
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d) and m.weight.grad is not None:
                    m.weight.grad.data.add_(
                        self._sparsity_rate * m.weight.data.sign()
                    )
            super().optimizer_step()

    return SparseTrainer


class YOLOv8Pruner:
    """
    基于 BN gamma L1 稀疏正则的 YOLOv8 通道剪枝器。

    完整流程:
        pruner = YOLOv8Pruner('yolov8n.pt', 'data/corn.yaml')
        base_pt    = pruner.base_train(epochs=200)            # (可选) 基础训练
        sparse_pt  = pruner.sparsity_train(epochs=100)        # 稀疏化训练
        pruner.analyze()                                       # 查看通道重要性
        pruned_pt  = pruner.prune(prune_ratio=0.25)           # 执行剪枝
        final_pt   = pruner.finetune(pruned_pt, epochs=50)    # 微调恢复精度
    """

    def __init__(self, model_path: str, data_yaml: str,
                 sparsity_rate: float = 0.02, device: str = ''):
        self.model_path    = model_path
        self.data_yaml     = data_yaml
        self.sparsity_rate = sparsity_rate
        self.device        = device
        self._sparse_model_path: Optional[str] = None

    # --------------------------------------------------------
    # Step 0: 基础训练 (可选)
    # --------------------------------------------------------

    def base_train(self, epochs: int = 200, batch: int = 16,
                   imgsz: int = 640, lr: float = 0.01,
                   project: str = 'runs/base',
                   name: str = 'train', **kwargs) -> str:
        """
        Step 0: 基础训练 YOLOv8（普通训练，不加稀疏正则）。

        用途:
            当你只有 yolov8n.pt / yolov8s.pt 等通用预训练权重、还没有在自己数据集上
            微调过的 best.pt 时，用此方法先做一次普通的迁移学习训练。
            训练完成后 self.model_path 会自动更新为新的 best.pt，
            后续 sparsity_train() 会自动从这个权重开始稀疏化。

        Args:
            epochs  : 训练轮数，论文实验常用 200
            batch   : batch size
            imgsz   : 输入尺寸
            lr      : 初始学习率（普通训练用 0.01 即可）
            project : 输出根目录
            name    : 子目录名
            **kwargs: 其余参数透传给 YOLO.train()

        Returns:
            训练完成后的 best.pt 绝对路径
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        print(f"\n[BaseTrain] 基础训练: {self.model_path}  "
              f"epochs={epochs}  batch={batch}  imgsz={imgsz}")
        YOLO(self.model_path).train(
            data=self.data_yaml, epochs=epochs, batch=batch,
            imgsz=imgsz, lr0=lr,
            project=project, name=name, device=self.device, **kwargs,
        )

        candidates = sorted(Path(project).glob(f'{name}*/weights/best.pt'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        best_pt = str(candidates[0]) if candidates \
                  else f'{project}/{name}/weights/best.pt'

        # 更新内部状态：后续稀疏化训练从这个权重开始
        self.model_path = best_pt
        print(f"[BaseTrain] 完成 → {best_pt}")
        print(f"[BaseTrain] self.model_path 已更新为该权重，下一步可直接调用 sparsity_train()")
        return best_pt

    # --------------------------------------------------------
    # Step 1: 稀疏化训练
    # --------------------------------------------------------

    def sparsity_train(self, epochs: int = 100, batch: int = 16,
                        imgsz: int = 640, project: str = 'runs/sparse',
                        name: str = 'train', **kwargs) -> str:
        """
        使用 L1 稀疏正则对模型进行稀疏化训练。

        论文参数: epochs=100, batch=16, sparsity_rate=0.02
        """
        SparseTrainer = make_sparse_trainer(self.sparsity_rate)
        SparseTrainer._sparsity_rate = self.sparsity_rate  # 支持实例级覆盖

        trainer = SparseTrainer(overrides=dict(
            model=self.model_path, data=self.data_yaml,
            epochs=epochs, batch=batch, imgsz=imgsz,
            project=project, name=name, device=self.device, **kwargs,
        ))
        print(f"[SparseTrainer] sparsity_rate={self.sparsity_rate}  epochs={epochs}")
        trainer.train()

        best_pt = Path(trainer.save_dir) / 'weights' / 'best.pt'
        self._sparse_model_path = str(best_pt)
        print(f"[SparseTrainer] 完成 → {best_pt}")
        return self._sparse_model_path

    # --------------------------------------------------------
    # Step 2: 分析通道重要性
    # --------------------------------------------------------

    def analyze(self, model=None, top_k: int = 20) -> Dict[str, torch.Tensor]:
        """分析各 BN 层 gamma 的分布，找出可剪通道。"""
        if model is None:
            model = self._load_model(self._sparse_model_path or self.model_path)
        nn_m = self._unwrap(model)

        importance = {
            name: m.weight.data.abs().clone().cpu()
            for name, m in nn_m.named_modules()
            if isinstance(m, nn.BatchNorm2d)
        }
        if not importance:
            print("[Analyze] 未找到 BN 层")
            return importance

        all_g = torch.cat(list(importance.values()))
        print(f"\n[Analyze] BN 层: {len(importance)}  |  gamma 统计:")
        for q in [0, 0.25, 0.5, 0.75, 1.0]:
            print(f"  p{int(q*100):3d} = {torch.quantile(all_g, q):.4f}")

        sorted_layers = sorted(importance.items(), key=lambda x: x[1].mean())[:top_k]
        print(f"\n  重要性最低的 {top_k} 层:")
        for i, (name, val) in enumerate(sorted_layers):
            print(f"  {i+1:2d}. {name:<55s}  mean={val.mean():.4f}  ch={val.numel()}")
        return importance

    # --------------------------------------------------------
    # Step 3: 剪枝
    # --------------------------------------------------------

    def prune(self, prune_ratio: float = 0.25,
               save_path: str = 'pruned_model.pt',
               model=None) -> str:
        """
        软剪枝: 将 |gamma| 最小的 prune_ratio 比例通道的 BN 参数清零。
        若已安装 torch-pruning (pip install torch-pruning)，同时执行硬剪枝。

        Args:
            prune_ratio : 剪枝比例，论文约 0.25
            save_path   : 剪枝后模型保存路径
        """
        if model is None:
            model = self._load_model(self._sparse_model_path or self.model_path)
        nn_m = self._unwrap(model)

        all_g = torch.cat([m.weight.data.abs().cpu().flatten()
                            for m in nn_m.modules()
                            if isinstance(m, nn.BatchNorm2d)])
        threshold = torch.quantile(all_g, prune_ratio).item()
        print(f"\n[Pruning] ratio={prune_ratio:.0%}  threshold={threshold:.6f}")

        masks: Dict[str, torch.Tensor] = {}
        total = kept = 0
        for name, m in nn_m.named_modules():
            if not isinstance(m, nn.BatchNorm2d):
                continue
            mask = m.weight.data.abs() >= threshold
            if mask.sum() == 0:                          # 每层至少保留 1 个通道
                mask[m.weight.data.abs().argmax()] = True
            masks[name] = mask
            m.weight.data[~mask] = 0.0
            m.bias.data[~mask]   = 0.0
            total += mask.numel()
            kept  += int(mask.sum())

        print(f"[Pruning] 软剪枝: 保留 {kept}/{total} ({100*kept/total:.1f}%)")

        if self._try_hard_prune(nn_m, masks):
            print("[Pruning] 硬剪枝成功，参数量已实际减少")
        else:
            print("[Pruning] 软剪枝完成（硬剪枝需 pip install torch-pruning）")

        params = sum(p.numel() for p in nn_m.parameters()) / 1e6
        print(f"[Pruning] 当前参数量: {params:.2f}M")

        self._save_model(model, save_path)
        print(f"[Pruning] 已保存 → {save_path}")
        return str(save_path)

    def _try_hard_prune(self, nn_m: nn.Module,
                         masks: Dict[str, torch.Tensor]) -> bool:
        """使用 torch-pruning 执行结构化硬剪枝（失败时静默返回 False）。"""
        try:
            import torch_pruning as tp   # pip install torch-pruning
        except ImportError:
            return False
        try:
            example = torch.randn(1, 3, 640, 640)
            DG = tp.DependencyGraph().build_dependency(nn_m, example_inputs=example)
            for name, m in nn_m.named_modules():
                if isinstance(m, nn.BatchNorm2d) and name in masks:
                    idxs = (~masks[name]).nonzero(as_tuple=True)[0].tolist()
                    if idxs:
                        group = DG.get_pruning_group(
                            m, tp.prune_batchnorm_out_channels, idxs=idxs)
                        if DG.check_pruning_group(group):
                            group.prune()
            return True
        except Exception as e:
            print(f"[Pruning] 硬剪枝出错 (忽略): {e}")
            return False

    # --------------------------------------------------------
    # Step 4: 微调
    # --------------------------------------------------------

    def finetune(self, pruned_model_path: str,
                  epochs: int = 50, batch: int = 16,
                  imgsz: int = 640, lr: float = 0.001,
                  project: str = 'runs/finetune',
                  name: str = 'finetune', **kwargs) -> str:
        """剪枝后微调，恢复精度。"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        print(f"\n[Finetune] 微调: {pruned_model_path}")
        result = YOLO(pruned_model_path).train(
            data=self.data_yaml, epochs=epochs, batch=batch,
            imgsz=imgsz, lr0=lr, lrf=lr * 0.01,
            project=project, name=name, device=self.device, **kwargs,
        )
        # Ultralytics 返回训练结果对象，save_dir 在 trainer 里
        # 用 glob 找最新生成的 best.pt
        candidates = sorted(Path(project).glob(f'{name}*/weights/best.pt'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        best_pt = str(candidates[0]) if candidates else f'{project}/{name}/weights/best.pt'
        print(f"[Finetune] 完成 → {best_pt}")
        return best_pt

    # --------------------------------------------------------
    # Step 5: CWD 蒸馏微调（可选）
    # --------------------------------------------------------

    def distill_finetune(self, teacher_path: str, student_path: str,
                          epochs: int = 100, batch: int = 16,
                          imgsz: int = 640, lr: float = 0.001,
                          cwd_weight: float = 1.0,
                          project: str = 'runs/distill',
                          name: str = 'distill') -> str:
        """
        用 CWD 通道知识蒸馏微调学生模型。

        修复说明:
          - 使用 preprocess_batch 缓存当前 batch 的图片张量
          - 在 optimizer_step 中先运行教师前向（触发特征钩子）
          - 用 scaler.scale(dist_loss).backward() 保证 AMP 兼容性
          - 用鸭子类型定位 C2f 模块，避免误钩 nn.Sequential
          - 使用 _setup_train 回调正确初始化教师模型
        """
        try:
            from ultralytics.models.yolo.detect.train import DetectionTrainer
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("pip install ultralytics")

        _cwd_loss   = CWDLoss(temperature=4.0)
        _cwd_weight = cwd_weight
        _teacher_path = teacher_path

        class DistillTrainer(DetectionTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._teacher: Optional[nn.Module] = None
                self._teacher_feats: List[torch.Tensor] = []
                self._student_feats: List[torch.Tensor] = []
                self._current_img:   Optional[torch.Tensor] = None

            # -- 教师初始化: 在训练开始时调用 --
            def _setup_train(self, world_size):
                super()._setup_train(world_size)
                self._init_teacher()

            def _init_teacher(self):
                """加载并冻结教师模型，注册 FPN 特征钩子。"""
                teacher_yolo = YOLO(_teacher_path)
                self._teacher = teacher_yolo.model.to(self.device).eval()
                for p in self._teacher.parameters():
                    p.requires_grad = False

                # 定位 C2f 类模块（鸭子类型: 有 cv1, cv2, m 属性）
                def is_c2f(m):
                    return (hasattr(m, 'cv1') and hasattr(m, 'cv2')
                            and hasattr(m, 'm') and isinstance(m.m, nn.ModuleList))

                from ultralytics.utils.torch_utils import de_parallel
                student_nn = de_parallel(self.model)

                t_c2f = [m for m in self._teacher.modules() if is_c2f(m)]
                s_c2f = [m for m in student_nn.modules()    if is_c2f(m)]

                # 钩住最后 3 个 C2f（对应 FPN 三个输出尺度）
                def make_hook(feat_list):
                    def hook(_, __, out):
                        if isinstance(out, torch.Tensor):
                            feat_list.append(out)
                    return hook

                for m in t_c2f[-3:]:
                    m.register_forward_hook(make_hook(self._teacher_feats))
                for m in s_c2f[-3:]:
                    m.register_forward_hook(make_hook(self._student_feats))

                print(f"[DistillTrainer] 教师加载完成，钩住 "
                      f"T:{len(t_c2f[-3:])} / S:{len(s_c2f[-3:])} 个 C2f 层")

                # 补丁: 将蒸馏损失合并进模型前向，与任务损失共享同一次 backward，
                # 彻底避免"backward through graph a second time"错误。
                _teacher_ref  = self._teacher
                _t_feats      = self._teacher_feats
                _s_feats      = self._student_feats
                _orig_forward = student_nn.forward   # 捕获原始 forward（已绑定 self）

                def _distill_forward(x, *args, **kwargs):
                    if isinstance(x, dict):          # 训练模式: 输入是 batch dict
                        _t_feats.clear()
                        _s_feats.clear()
                        with torch.no_grad():
                            _teacher_ref(x['img'])   # 教师前向，填充 _t_feats
                        out = _orig_forward(x, *args, **kwargs)  # 学生前向，填充 _s_feats
                        n = min(len(_s_feats), len(_t_feats))
                        if n > 0:
                            d_loss = sum(
                                _cwd_loss(sf, tf)
                                for sf, tf in zip(_s_feats[:n], _t_feats[:n])
                                if sf.shape == tf.shape
                            )
                            if isinstance(d_loss, torch.Tensor):
                                out = (out[0] + _cwd_weight * d_loss,) + out[1:]
                        _t_feats.clear()
                        _s_feats.clear()
                        return out
                    return _orig_forward(x, *args, **kwargs)  # 推理/验证模式不变

                student_nn.forward = _distill_forward

            # 蒸馏损失已在 _distill_forward 内合并进任务损失，
            # optimizer_step 无需额外处理，走标准流程即可。

        trainer = DistillTrainer(overrides=dict(
            model=student_path, data=self.data_yaml,
            epochs=epochs, batch=batch, imgsz=imgsz,
            lr0=lr, project=project, name=name, device=self.device,
        ))
        trainer.train()

        candidates = sorted(Path(project).glob(f'{name}*/weights/best.pt'),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        best_pt = str(candidates[0]) if candidates else f'{project}/{name}/weights/best.pt'
        print(f"[DistillTrainer] 完成 → {best_pt}")
        return best_pt

    # --------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------

    @staticmethod
    def _load_model(path: str):
        try:
            from ultralytics import YOLO
            return YOLO(path)
        except ImportError:
            raise ImportError("pip install ultralytics")

    @staticmethod
    def _unwrap(model) -> nn.Module:
        """从 YOLO 包装对象或 nn.Module 中取出底层 nn.Module。"""
        if isinstance(model, nn.Module):
            return model
        if hasattr(model, 'model'):
            m = model.model
            return m.model if hasattr(m, 'model') else m
        raise ValueError(f"无法提取 nn.Module: {type(model)}")

    @staticmethod
    def _save_model(model, path: str):
        path = str(path)
        try:
            if hasattr(model, 'save'):
                model.save(path)
            else:
                torch.save({'model': model}, path)
        except Exception as e:
            print(f"[Warning] 保存时出错: {e}")
            torch.save({'model': model}, path)

    def model_stats(self, model=None) -> dict:
        """打印参数量与 BN gamma 稀疏度统计。"""
        if model is None:
            model = self._load_model(self._sparse_model_path or self.model_path)
        nn_m  = self._unwrap(model)
        params = sum(p.numel() for p in nn_m.parameters()) / 1e6
        gammas = torch.cat([m.weight.data.abs().cpu().flatten()
                             for m in nn_m.modules()
                             if isinstance(m, nn.BatchNorm2d)])
        near0  = (gammas < 0.01).float().mean().item() * 100
        stats  = {'params_M': params, 'bn_channels': gammas.numel(), 'near_zero_pct': near0}
        print(f"[Stats] {params:.2f}M 参数 | {gammas.numel()} BN 通道 | {near0:.1f}% 近零")
        return stats


# ============================================================
# 第三部分: 一键剪枝入口
# ============================================================

def prune_yolov8(
    model_path: str,
    data_yaml: str,
    sparsity_rate: float   = 0.02,
    prune_ratio: float     = 0.25,
    base_train_epochs: int = 0,      # 0 = 跳过基础训练；>0 = 先做 Step 0
    sparse_epochs: int     = 100,
    finetune_epochs: int   = 50,
    batch: int   = 16,
    imgsz: int   = 640,
    device: str  = '',
    output_dir: str = 'runs/pruning',
    do_distill: bool = False,
) -> dict:
    """
    完整剪枝流程 (论文方法)。

    Args:
        model_path         : 起始权重
                                - base_train_epochs > 0 时: 用 'yolov8n.pt' 等预训练权重
                                - base_train_epochs = 0 时: 用你已训练好的 'best.pt'
        data_yaml          : 数据集配置文件路径，或由 DatasetConfig.generate_yaml() 生成
        sparsity_rate      : L1 稀疏正则系数，论文选 0.02
        prune_ratio        : 剪枝比例，论文约 0.25
        base_train_epochs  : Step 0 基础训练轮数（0 表示跳过，用已有 best.pt）
        sparse_epochs      : Step 1 稀疏训练轮数
        finetune_epochs    : Step 3 微调轮数
        batch / imgsz      : 训练参数
        device             : '' 自动 / 'cpu' / '0' 单 GPU
        output_dir         : 输出根目录
        do_distill         : 是否额外做 Step 4 CWD 蒸馏微调

    Returns:
        各阶段权重路径字典，可能包含键:
            'base_model'   (仅当 base_train_epochs > 0)
            'sparse_model'
            'pruned_model'
            'final_model'
    """
    os.makedirs(output_dir, exist_ok=True)
    pruner = YOLOv8Pruner(model_path, data_yaml,
                           sparsity_rate=sparsity_rate, device=device)
    result: dict = {}

    total_steps = 3 + (1 if base_train_epochs > 0 else 0) + (1 if do_distill else 0)

    print("=" * 60)
    print("YOLOv8 剪枝流程")
    print(f"  起始权重:   {model_path}")
    print(f"  数据集:     {data_yaml}")
    print(f"  稀疏率:     {sparsity_rate}")
    print(f"  剪枝比例:   {prune_ratio:.0%}")
    print(f"  基础训练:   {'跳过' if base_train_epochs == 0 else f'{base_train_epochs} epochs'}")
    print(f"  蒸馏微调:   {'启用' if do_distill else '跳过'}")
    print(f"  总步骤数:   {total_steps}")
    print("=" * 60)

    # ---- Step 0 (可选): 基础训练 ----
    if base_train_epochs > 0:
        print(f"\n▶ Step 0  基础训练 ({base_train_epochs} epochs)")
        base_pt = pruner.base_train(
            epochs=base_train_epochs, batch=batch, imgsz=imgsz,
            project=f'{output_dir}/base', name='train')
        result['base_model'] = base_pt
        # 注意: pruner.model_path 已在 base_train() 内部更新

    # ---- Step 1: 稀疏化训练 ----
    print(f"\n▶ Step 1  稀疏化训练 ({sparse_epochs} epochs)")
    sparse_pt = pruner.sparsity_train(
        epochs=sparse_epochs, batch=batch, imgsz=imgsz,
        project=f'{output_dir}/sparse', name='train')
    result['sparse_model'] = sparse_pt

    # ---- Step 2: 剪枝 ----
    print("\n▶ Step 2  通道剪枝")
    pruner.analyze()
    pruned_pt = pruner.prune(
        prune_ratio=prune_ratio,
        save_path=f'{output_dir}/pruned.pt')
    result['pruned_model'] = pruned_pt

    # ---- Step 3: 微调 ----
    print(f"\n▶ Step 3  回调微调 ({finetune_epochs} epochs)")
    final_pt = pruner.finetune(
        pruned_pt, epochs=finetune_epochs, batch=batch, imgsz=imgsz,
        project=f'{output_dir}/finetune', name='finetune')
    result['final_model'] = final_pt

    # ---- Step 4 (可选): CWD 蒸馏 ----
    if do_distill:
        print(f"\n▶ Step 4  CWD 蒸馏微调 ({finetune_epochs} epochs)")
        distill_pt = pruner.distill_finetune(
            teacher_path=sparse_pt, student_path=pruned_pt,
            epochs=finetune_epochs, batch=batch, imgsz=imgsz,
            project=f'{output_dir}/distill', name='distill')
        result['final_model'] = distill_pt

    print("\n" + "=" * 60)
    print("✓ 完成!")
    for k, v in result.items():
        print(f"  {k:<15s}: {v}")
    print("=" * 60)
    return result


# ============================================================
# Demo
# ============================================================

def demo():
    print("=" * 62)
    print("YOLOv8-Corn 模块验证 + 剪枝工具演示")
    print("=" * 62)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    B = 2

    # ---- 算法模块 ----
    print("① DCNv2")
    x = torch.randn(B, 64, 80, 80, device=device)
    print(f"   {tuple(x.shape)} → {tuple(DCNv2(64,128,3,1,1).to(device)(x).shape)}")

    print("\n② C2f-DCNv2")
    x = torch.randn(B, 64, 80, 80, device=device)
    print(f"   {tuple(x.shape)} → {tuple(C2fDCNv2(64,64,n=2).to(device)(x).shape)}")

    print("\n③ DySample ×2")
    x = torch.randn(B, 128, 20, 20, device=device)
    print(f"   {tuple(x.shape)} → {tuple(DySample(128,2).to(device)(x).shape)}")

    print("\n④ CWDLoss")
    fs = torch.randn(B, 64, 40, 40, device=device)
    ft = torch.randn(B, 64, 40, 40, device=device)
    print(f"   L_CWD = {CWDLoss(4.0)(fs, ft).item():.4f}")

    print("\n⑤ MGDLoss")
    fs = torch.randn(B, 32, 40, 40, device=device)
    ft = torch.randn(B, 64, 40, 40, device=device)
    print(f"   L_MGD = {MGDLoss(32,64,0.5).to(device)(fs,ft).item():.4f}")

    # ---- DatasetConfig ----
    print("\n⑥ DatasetConfig")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        # 创建一个最小数据集目录结构用于演示
        for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        # 放一张假图片和对应标注
        open(os.path.join(tmp, 'images/train/img1.jpg'), 'w').close()
        with open(os.path.join(tmp, 'labels/train/img1.txt'), 'w') as f:
            f.write('0 0.5 0.5 0.3 0.3\n')
        open(os.path.join(tmp, 'images/val/img2.jpg'), 'w').close()
        with open(os.path.join(tmp, 'labels/val/img2.txt'), 'w') as f:
            f.write('1 0.4 0.4 0.2 0.2\n')

        cfg = DatasetConfig(tmp, ['leaf_spot', 'gray_spot', 'mosaic', 'rust'])
        cfg.check()
        yaml_path = cfg.generate_yaml()
        print(f"   生成的 yaml: {yaml_path}")
        print(f"   内容:\n" + Path(yaml_path).read_text(encoding='utf-8'))

    # ---- 使用说明 ----
    print("=" * 62)
    print("【使用方式】\n")
    print("""\
# ── 方式 A: 从预训练权重开始，一条龙跑完 Step 0→3 ─────────
#     (你还没有自己训练过的 best.pt，只有官方 yolov8n.pt)
from yolov8_corn import prune_yolov8

prune_yolov8(
    model_path        = 'yolov8n.pt',     # 官方预训练权重
    data_yaml         = 'C:/mydata/data.yaml',
    base_train_epochs = 200,              # ← Step 0: 在你的数据集上先训 200 轮
    sparse_epochs     = 100,              # Step 1
    prune_ratio       = 0.25,             # Step 2
    finetune_epochs   = 50,               # Step 3
    sparsity_rate     = 0.02,
    batch   = 16,
    imgsz   = 640,
    device  = '0',
    do_distill = True,                    # Step 4 (可选)
)

# ── 方式 B: 已有 best.pt，跳过 Step 0 ────────────────────
from yolov8_corn import prune_yolov8

prune_yolov8(
    model_path        = 'runs/detect/train/weights/best.pt',
    data_yaml         = 'C:/mydata/data.yaml',
    base_train_epochs = 0,                # ← 0 表示跳过基础训练
    sparse_epochs     = 100,
    prune_ratio       = 0.25,
    finetune_epochs   = 50,
    batch   = 16,
    imgsz   = 640,
    device  = '0',
)

# ── 方式 C: 自动生成 data.yaml + 完整流程 ───────────────
from yolov8_corn import DatasetConfig, prune_yolov8

cfg = DatasetConfig(
    dataset_dir = 'D:/mydata',
    class_names = ['leaf_spot', 'gray_spot', 'mosaic', 'rust'],
)
cfg.check()
data_yaml = cfg.generate_yaml()

prune_yolov8(
    model_path        = 'yolov8n.pt',
    data_yaml         = data_yaml,
    base_train_epochs = 200,
)

# ── 方式 D: 手动分步调用（更灵活）────────────────────────
from yolov8_corn import YOLOv8Pruner

pruner = YOLOv8Pruner('yolov8n.pt', 'C:/mydata/data.yaml',
                       sparsity_rate=0.02, device='0')

base_pt   = pruner.base_train(epochs=200)          # Step 0
sparse_pt = pruner.sparsity_train(epochs=100)      # Step 1 (自动用 base_pt)
pruner.analyze()                                    # 查看通道重要性
pruned_pt = pruner.prune(prune_ratio=0.25)         # Step 2
final_pt  = pruner.finetune(pruned_pt, epochs=50)  # Step 3
""")
    print("✓ 演示完成！")


if __name__ == '__main__':
    demo()