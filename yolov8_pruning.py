# This code is adapted for YOLO11 (only C3k2 support)
import argparse
import math
import os
import gc

os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # 使用离屏渲染，不需要GUI
os.environ['OMP_NUM_THREADS'] = '1'  # 限制CPU线程数，减少资源竞争
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import logging
import warnings

# 设置Qt环境变量，避免GUI相关错误
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
os.environ['QT_SCALE_FACTOR'] = '1'
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
# ========== 只保留C3k2导入 ==========
from ultralytics.nn.modules import Detect, C3k2, Conv, Bottleneck
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator, DetectionPredictor

TASK_MAP = {
    "detect": [
        "yolo11n.pt",  # YOLO11默认模型
        DetectionTrainer,
        DetectionValidator,
        DetectionPredictor
    ]
}

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights, de_parallel

import torch_pruning as tp
import yaml  # 直接用PyYAML库

# ========== 全局配置：限制资源使用 ==========
# torch.set_num_threads(1)  # 限制PyTorch线程数
# torch.backends.cudnn.benchmark = False  # 关闭cudnn基准测试，减少内存占用
# torch.backends.cudnn.deterministic = True



# 手动实现yaml_load函数
def yaml_load(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_pruning_performance_graph(x, y1, y2, y3):
    """
    Draw performance change graph
    Parameters
    ----------
    x : List
        Parameter numbers of all pruning steps
    y1 : List
        mAPs after fine-tuning of all pruning steps
    y2 : List
        MACs of all pruning steps
    y3 : List
        mAPs after pruning (not fine-tuned) of all pruning steps
    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig('pruning_perf_change.png')
    # 释放matplotlib资源
    plt.close(fig)
    gc.collect()


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


def transfer_weights_c3k(src_module, dst_module):
    """权重迁移 + Ultralytics 属性兼容"""
    dst_module.m = src_module.m
    dst_module.cv2 = src_module.cv2

    # 复制 Ultralytics 需要的属性
    for attr in ['f', 'i', 'type']:
        if hasattr(src_module, attr):
            setattr(dst_module, attr, getattr(src_module, attr))

    # ...（原来的权重迁移代码保持不变）...

    state_dict = src_module.state_dict()
    dst_state = dst_module.state_dict()

    if 'cv1.conv.weight' in state_dict:
        old_w = state_dict['cv1.conv.weight']
        half = old_w.shape[0] // 2
        dst_state['cv0.conv.weight'] = old_w[:half]
        dst_state['cv1.conv.weight'] = old_w[half:]

        for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
            key = f'cv1.bn.{bn_key}'
            if key in state_dict:
                old_bn = state_dict[key]
                dst_state[f'cv0.bn.{bn_key}'] = old_bn[:half]
                dst_state[f'cv1.bn.{bn_key}'] = old_bn[half:]

    for k, v in state_dict.items():
        if k in dst_state and not k.startswith('cv1.'):
            dst_state[k] = v

    dst_module.load_state_dict(dst_state, strict=False)


def save_model_v2(self: BaseTrainer):
    """Disabled half precision saving"""
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt
    gc.collect()  # 释放内存


def final_eval_v2(self: BaseTrainer):
    """Final evaluation with stripped optimizer"""
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """Disabled half precision saving"""
    x = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    args = {**DEFAULT_CFG_DICT, **x['train_args']}
    if x.get('ema'):
        x['model'] = x['ema']
    for k in 'optimizer', 'ema', 'updates':
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    # 释放内存
    del x, args, p, mb
    gc.collect()



def train_v2(self: YOLO, pruning=False, **kwargs):
    """已修复：增加全参数支持的融合拦截逻辑 (YOLO11适配)"""
    if self.session:
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    overrides["amp"] = False

    # 1. 准备参数与 Trainer
    if 'model' not in overrides or overrides['model'] is None:
        if hasattr(self, 'ckpt_path') and self.ckpt_path is not None:
            overrides['model'] = str(self.ckpt_path)
        else:
            overrides['model'] = "yolo11n.pt"

    if kwargs.get('cfg'):
        overrides = yaml_load(check_yaml(kwargs['cfg']))
        
    overrides['mode'] = 'train'
    self.task = overrides.get('task') or self.task
    self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)

    # 2. 区分剪枝模式与标准模式
    if not pruning:
        if not overrides.get('resume'):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
    else:
        # --- 核心修改：激活全局融合拦截 (支持所有传入参数) ---
        print("\n" + "=" * 60)
        print("🛡️  剪枝模式：激活全局融合拦截 (Global Anti-Fuse)")
        print("=" * 60)

        self.trainer.pruning = True
        self.trainer.model = self.model

        # 拦截 1: 禁用所有模块的融合前向路径
        for m in self.trainer.model.modules():
            if hasattr(m, "forward_fuse"):
                m.forward = m.forward

        # 拦截 2: 修改类行为，使用可变参数以支持 verbose 等参数
        import ultralytics.nn.tasks as tasks
        # 必须带 *args 和 **kwargs 来接收 Ultralytics 传进来的 verbose 等参数
        tasks.DetectionModel.fuse = lambda self, *args, **kwargs: self
        if hasattr(tasks, 'SegmentationModel'):
            tasks.SegmentationModel.fuse = lambda self, *args, **kwargs: self

        # 3. 设备一致性修复 (确保在 else 分支内)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        self.trainer.model = self.trainer.model.to(device)

        def recursive_to_device(module, target_device):
            for param in module.parameters(recurse=False):
                if param.device != target_device:
                    module.to(target_device)
                    break
            for child in module.children():
                recursive_to_device(child, target_device)

        recursive_to_device(self.trainer.model, device)

        # 4. 修复损失函数
        if hasattr(self.trainer.model, 'criterion'):
            criterion = self.trainer.model.criterion
            for attr_name in dir(criterion):
                try:
                    attr = getattr(criterion, attr_name)
                    if isinstance(attr, torch.Tensor) and attr.device != device:
                        setattr(criterion, attr_name, attr.to(device))
                except: continue
            if hasattr(criterion, 'proj') and criterion.proj.device != device:
                criterion.proj = criterion.proj.to(device)

        # 5. 替换保存与评估逻辑
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    # 6. 启动训练 (确保缩进正确，pruning 或不 pruning 都要执行)
    self.trainer.hub_session = self.session
    self.trainer.train()

    # 7. 训练结束后清理
    if RANK in (-1, 0):
        # 再次确认拦截，防止加载权重时的自动融合
        import ultralytics.nn.tasks as tasks
        tasks.DetectionModel.fuse = lambda self, *args, **kwargs: self
        
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PruningLogFilter(logging.Filter):
    """过滤掉训练过程中烦人的日志"""

    def filter(self, record):
        bad_words = ["requires_grad", "frozen layer", "optimizer", "AMP", "torch.meshgrid"]
        return not any(word in record.getMessage().lower() for word in bad_words)


class AdaptiveConv2d(nn.Module):
    """
    自适应卷积包装层 (补齐版)：
    透传所有卷积属性以欺骗 Ultralytics 的 fuse() 机制。
    """

    def __init__(self, conv_layer):
        super().__init__()
        self.conv = conv_layer

    # --- 必须透传的属性，防止 AttributeError ---
    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def kernel_size(self):
        return self.conv.kernel_size

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    @property
    def dilation(self):
        return self.conv.dilation

    @property
    def groups(self):
        return self.conv.groups

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    @property
    def padding_mode(self):
        return self.conv.padding_mode

    @property
    def trans(self):
        return getattr(self.conv, 'trans', False)  # 兼容某些特殊层

    def forward(self, x):
        actual_in_c = x.shape[1]
        target_in_c = self.conv.weight.shape[1] * self.conv.groups

        if actual_in_c > target_in_c:
            x = x[:, :target_in_c, :, :]
        elif actual_in_c < target_in_c:
            pad_shape = list(x.shape)
            pad_shape[1] = target_in_c - actual_in_c
            padding = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)

        return self.conv(x)


def check_model_channels(model, stage=""):
    """严格通道一致性检查"""
    print(f"\n🔍 === 通道一致性检查 [{stage}] ===")
    error_count = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            weight_in = m.weight.shape[1]  # 模块的输入通道
            declared_in = m.in_channels
            weight_out = m.weight.shape[0]  # 模块的输出通道
            declared_out = m.out_channels

            if weight_in != declared_in:
                print(f"❌ 输入不匹配: {name:50} weight_in={weight_in} declared={declared_in}")
                error_count += 1
            if weight_out != declared_out:
                print(f"❌ 输出不匹配: {name:50} weight_out={weight_out} declared={declared_out}")
                error_count += 1
    if error_count == 0:
        print("✅ 所有 Conv 层通道完全一致！结构安全")
    else:
        print(f"⚠️  共发现 {error_count} 处通道不匹配！")
    print("=" * 70)
    return error_count == 0


class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            return y[0]
        return y


def yolo11_trace_forward(model, x):
    """确保在 trace 期间始终能捕获到有效的 Tensor 路径"""
    # 确保输入有梯度
    if not x.requires_grad:
        x.requires_grad_(True)

    results = model(x)

    # 递归提取所有带 grad_fn 的 Tensor
    def extract_tensors(obj):
        tensors = []
        if isinstance(obj, torch.Tensor):
            if obj.grad_fn is not None:
                tensors.append(obj)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                tensors.extend(extract_tensors(item))
        elif isinstance(obj, dict):
            for v in obj.values():
                tensors.extend(extract_tensors(v))
        return tensors

    valid_outputs = extract_tensors(results)

    # 如果实在没抓到（比如模型全是推理层），返回原始结果
    return valid_outputs if valid_outputs else results


def get_mha_norm_indices(layer, prune_ratio, num_heads=2):
    out_c = layer.out_channels  # 256

    # 1. 逻辑分段：找到能被 3 整除的基数 (255)
    c_base = (out_c // 3) * 3  # 255
    c_part = c_base // 3  # 85

    w = layer.weight.data.detach().abs().mean(dim=(1, 2, 3))

    # 2. 对称得分计算
    q_s, k_s, v_s = w[0:85], w[85:170], w[170:255]
    combined_scores = q_s + k_s + v_s

    # 3. 计算保留数量：必须能被 num_heads 整除
    # 假设保留 70%，59.5 -> 58 (对齐到 2 heads)
    expected_keep = int(c_part * (1 - prune_ratio))
    aligned_keep = (expected_keep // num_heads) * num_heads
    prune_per_part = c_part - aligned_keep

    # 4. 获取 QKV 的剪枝索引
    _, rel_idx = torch.topk(combined_scores, prune_per_part, largest=False)

    final_idx = []
    for i in range(3):
        final_idx.extend((rel_idx + i * c_part).tolist())

    # 5. 【关键】将那个多出来的第 256 位 (索引 255) 强制加入剪枝列表
    if out_c > c_base:
        for extra_i in range(c_base, out_c):
            final_idx.append(extra_i)

    return sorted(list(set(final_idx)))


def final_physical_align(model):
    print("🚀 正在执行物理属性强制纠正...")
    for name, m in model.named_modules():
        # 如果使用了包装层，先拿到原始卷积
        curr_m = m.conv if isinstance(m, AdaptiveConv2d) else m

        if isinstance(curr_m, nn.Conv2d):
            # 纠正输出通道
            if curr_m.out_channels != curr_m.weight.shape[0]:
                curr_m.out_channels = curr_m.weight.shape[0]

            # 纠正输入通道 (考虑 groups)
            expected_in = curr_m.weight.shape[1] * curr_m.groups
            if curr_m.in_channels != expected_in:
                curr_m.in_channels = expected_in

        elif isinstance(curr_m, nn.BatchNorm2d):
            if curr_m.num_features != curr_m.weight.shape[0]:
                curr_m.num_features = curr_m.weight.shape[0]
    return model


def report_detailed_physical_channels(model, title="剪枝后模型物理状态报告"):
    import pandas as pd  # 如果没有 pandas，用 print 格式化也可以

    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(f"{'Module Name':<40} | {'Logic (In/Out)':<15} | {'Physical Weight Shape':<20} | {'Groups':<6}")
    print("-" * 90)

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # 提取逻辑声明
            if isinstance(m, nn.Conv2d):
                logic_in, logic_out = m.in_channels, m.out_channels
                phys_shape = list(m.weight.shape)  # [out, in/groups, k, k]
                groups = m.groups
            else:
                logic_in, logic_out = m.in_features, m.out_features
                phys_shape = list(m.weight.shape)  # [out, in]
                groups = "-"

            # 检查是否存在不匹配 (红色标记逻辑)
            error_flag = ""
            if isinstance(m, nn.Conv2d):
                # 深度卷积逻辑：in == groups, 物理输入恒为 1
                if groups > 1:
                    if logic_in != groups: error_flag = "❌ Group mismatch"
                else:
                    if phys_shape[1] != logic_in: error_flag = "❌ Input mismatch"

                if phys_shape[0] != logic_out:
                    error_flag += " ❌ Output mismatch"

            print(
                f"{name[:40]:<40} | {logic_in:>3}/{logic_out:>3}       | {str(phys_shape):<20} | {groups:<6} {error_flag}")

        elif isinstance(m, nn.BatchNorm2d):
            logic_feat = m.num_features
            phys_weight = m.weight.shape[0]
            error_flag = "❌ BN mismatch" if logic_feat != phys_weight else ""
            print(f"{name[:40]:<40} | -- /{logic_feat:>3}       | weight:[{phys_weight}]        | -      {error_flag}")

    print("-" * 90)


def prune_c3k2_custom(model, layer_name, module):
    """
    针对 C3k2 模块的增强版物理对齐补丁
    """
    import torch
    import torch.nn as nn

    with torch.no_grad():
        # 1. 找到核心组件
        cv1_conv = module.cv1.conv
        cv1_bn = module.cv1.bn

        # 2. 确定当前物理通道数 (必须为偶数)
        actual_channels = cv1_conv.weight.shape[0]
        if actual_channels % 2 != 0:
            actual_channels -= 1
            # 强制对齐物理权重
            cv1_conv.weight.data = cv1_conv.weight.data[:actual_channels].clone()
            if cv1_conv.bias is not None:
                cv1_conv.bias.data = cv1_conv.bias.data[:actual_channels].clone()

        # 更新逻辑属性
        cv1_conv.out_channels = actual_channels
        module.cv1.out_channels = actual_channels  # Ultralytics 包装类属性

        # 3. 强制裁剪 BatchNorm (你原本的逻辑，很好)
        if isinstance(cv1_bn, nn.BatchNorm2d):
            cv1_bn.num_features = actual_channels
            for attr in ['running_mean', 'running_var']:
                tensor = getattr(cv1_bn, attr)
                if tensor.shape[0] != actual_channels:
                    setattr(cv1_bn, attr, tensor[:actual_channels].clone())

            if cv1_bn.weight is not None:
                cv1_bn.weight.data = cv1_bn.weight.data[:actual_channels].clone()
            if cv1_bn.bias is not None:
                cv1_bn.bias.data = cv1_bn.bias.data[:actual_channels].clone()

        # 4. 更新 Split 核心索引 c
        new_c = actual_channels // 2
        module.c = new_c

        # 5. 同步内部 Bottleneck 链
        if hasattr(module, 'm'):
            for b in module.m:
                # 修复 Bottleneck 的 cv1 输入端
                b_cv1 = b.cv1.conv
                b_cv1.weight.data = b_cv1.weight.data[:, :new_c, :, :].clone()
                b_cv1.in_channels = new_c
                b.cv1.in_channels = new_c  # 包装类同步

        # 6. 【新增】修复末端 cv2 的输入端
        # C3k2 的 concat 数量通常是 len(m) + 1
        if hasattr(module, 'cv2'):
            m_len = len(module.m) if hasattr(module, 'm') else 0
            cv2_in = (m_len + 1) * new_c

            module.cv2.conv.in_channels = cv2_in
            module.cv2.in_channels = cv2_in

            # 物理裁剪权重 dim=1 (输入通道维度)
            old_cv2_w = module.cv2.conv.weight.data
            if old_cv2_w.shape[1] != cv2_in:
                module.cv2.conv.weight.data = old_cv2_w[:, :cv2_in, :, :].clone()

        print(
            f"✨ {layer_name} 物理对齐成功: Out={actual_channels}, Split_c={new_c}, CV2_In={cv2_in if hasattr(module, 'cv2') else 'N/A'}")


def force_bn_stats_align(model):
    print("🛠️ 正在执行 BN 统计量物理强行对齐...")
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            # 以物理权重 (weight) 的形状为基准，因为这是 tp 肯定会修改的地方
            actual_channels = m.weight.data.shape[0]

            # 强制同步逻辑属性
            m.num_features = actual_channels

            # 关键：使用正确的 buffer 裁剪方式
            # 必须检查所有潜在的 buffer
            for attr in ['running_mean', 'running_var']:
                buffer = getattr(m, attr)
                if buffer is not None and buffer.shape[0] != actual_channels:
                    # 使用 .narrow 或切片，并确保类型一致
                    new_data = buffer.data[:actual_channels].clone()
                    setattr(m, attr, new_data)

            # 处理可能的 bias
            if m.bias is not None and m.bias.data.shape[0] != actual_channels:
                m.bias.data = m.bias.data[:actual_channels].clone()

            count += 1
    return model


import types


# ==================== [ 新增：特征图迎合补丁函数 ] ====================

def make_layer_adaptive(module):
    """让卷积层具备‘特征图迎合’能力：自动裁切或补齐输入通道"""
    if not isinstance(module, nn.Conv2d):
        return

    original_forward = module.forward

    # --- 在文件全局作用域定义这个类 ---


def adaptive_c3k2_forward(self, x):
    """C3k2 专用补丁：让分割操作动态迎合 cv1 的输出特征图"""
    y = self.cv1(x)
    current_c = y.shape[1]
    mid = current_c // 2
    y1, y2 = y.split((mid, current_c - mid), 1)
    return self.cv2(torch.cat((y1, y2, *(m(y2) for m in self.m)), 1))


# =====================================================================

def safe_prune_model(yolo_container, pruning_ratio, example_inputs, target_device):
    """
    针对 YOLO11 优化的安全剪枝函数 (手动物理对齐 + 特征图动态迎合版)
    """
    print(f"\n🔧 启动剪枝流程 (目标比例: {pruning_ratio})...")
    torch.set_grad_enabled(True)

    # 核心：提取底层的 DetectionModel
    model = yolo_container.model.cpu()

    # ========== [ 1. 注入动态补丁 (防止构建依赖图时报错) ] ==========
    # for n, m in model.named_modules():
    #     if "C3k2" in str(type(m)):
    #         m.forward = types.MethodType(adaptive_c3k2_forward, m)
    #         if hasattr(m, 'cv1'):
    #             c_out = m.cv1.conv.out_channels if hasattr(m.cv1, 'conv') else m.cv1.out_channels
    #             mid = c_out // 2
    #             # 必须给 module 增加这个属性，TP 的索引映射器才能读到它
    #             m.split_sizes = [mid, c_out - mid]
    # ========== [ 2. 注入动态 Shape 补丁 (针对 Attention) ] ==========
    target_attn_name = "10.m.0.attn"
    target_attn = next((m for n, m in model.named_modules() if target_attn_name in n and hasattr(m, 'qkv')), None)

    if target_attn:
        print(f"💉 正在注入动态 Shape 补丁: {target_attn_name}")
        original_attn_forward = target_attn.forward

        def patched_forward(x):
            qkv = target_attn.qkv(x)
            B, C, H, W = qkv.shape
            new_c_per_head = C // target_attn.num_heads
            qkv_reshaped = qkv.view(B, target_attn.num_heads, new_c_per_head, -1)
            virtual_feature = qkv_reshaped.mean(dim=1).view(B, -1, H, W)
            return target_attn.proj(virtual_feature)

        target_attn.forward = patched_forward

    # ========== [ 3. 构建依赖图 ] ==========
    example_inputs = example_inputs.cpu().detach().requires_grad_(True)
    shortcut_backups = {}
    print("🔓 正在逻辑断开残差连接以解锁普通层...")
    for name, m in model.named_modules():
        if hasattr(m, 'add') and getattr(m, 'shortcut', False):
            shortcut_backups[name] = m.add
            m.add = False

    DG = tp.DependencyGraph()
    ignored_layers = [m for m in model.modules() if isinstance(m, (Detect,))]
    DG.build_dependency(model, example_inputs, forward_fn=yolo11_trace_forward, ignored_layers=ignored_layers)

    for name, original_state in shortcut_backups.items():
        dict(model.named_modules())[name].add = original_state
    print("🔒 残差连接已恢复，依赖图构建完成。")

    # ========== [ 4. 筛选可剪枝层 ] ==========
    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if any(x in name.lower() for x in ["detect", "dfl", "model.10", "attn", "psa"]): continue
            if module.groups > 1: continue
            if module.out_channels <= 1: continue
            prunable_layers.append((name, module))

    # ========== [ 5. 执行剪枝循环 ] ==========
    processed_modules = set()
    for name, layer in prunable_layers:
        parent_name = ".".join(name.split('.')[:-2])
        parent_module = dict(model.named_modules()).get(parent_name)

        if isinstance(parent_module, C3k2):
            if parent_name in processed_modules: continue
            try:
                target_layer = parent_module.cv1.conv
                out_channels = target_layer.out_channels
                calc_prune_num = int(out_channels * pruning_ratio)
                remaining = out_channels - calc_prune_num
                aligned_remaining = max(8, (remaining // 2) * 2)
                prune_num = out_channels - aligned_remaining

                if prune_num <= 0:
                    processed_modules.add(parent_name)
                    continue

                score = target_layer.weight.detach().abs().mean(dim=(1, 2, 3))
                prune_idx = torch.topk(score, prune_num, largest=False).indices
                group = DG.get_pruning_group(target_layer, tp.prune_conv_out_channels, idxs=prune_idx)

                if DG.check_pruning_group(group):
                    group.exec()
                    prune_c3k2_custom(model, parent_name, parent_module)
                    processed_modules.add(parent_name)
                    print(f"🔥 [C3k2] 逻辑剪枝并执行专项对齐完成: {parent_name}")
                continue
            except Exception as e:
                print(f"⚠️ C3k2 异常 {parent_name}: {e}")
                continue

        if any(p_name in name for p_name in processed_modules): continue

        try:
            out_channels = layer.out_channels
            prune_num = int(out_channels * pruning_ratio)
            if prune_num <= 0 or (out_channels - prune_num) < 4: continue

            score = layer.weight.detach().abs().mean(dim=(1, 2, 3))
            prune_idx = torch.topk(score, prune_num, largest=False).indices
            group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=prune_idx)

            if DG.check_pruning_group(group):
                group.exec()
                # 显式更新逻辑属性
                for dep, _ in group:
                    m = dep.target.module
                    if isinstance(m, nn.BatchNorm2d):
                        m.num_features = m.weight.shape[0]
                    elif isinstance(m, (nn.Conv2d, nn.Linear)):
                        if hasattr(m, 'weight'):
                            m.out_channels = m.weight.shape[0]
                            m.in_channels = m.weight.shape[1]
                print(f"✅ 成功剪枝: {name}")

        except Exception as e:
            print(f"⚠️ 跳过 {name} | {e}")

    # # ========== [ 6. 最终补丁：实施“特征图迎合模块” ] ==========
    print("🩹 正在注入全局 AdaptiveConv2d 适配器 (防递归版)...")

    # 1. 先把需要替换的任务记录下来，不要在遍历过程中直接修改模型结构
    replace_list = []

    for name, m in model.named_modules():
        for child_name, child_module in m.named_children():
            # 检查是否是卷积层，且【不是】已经包装过的适配层
            if isinstance(child_module, nn.Conv2d) and not isinstance(child_module, AdaptiveConv2d):
                # 排除检测头
                if any(key in f"{name}.{child_name}" for key in ["detect", "dfl"]):
                    continue
                # 记录：父模块、子模块名、原始模块
                replace_list.append((m, child_name, child_module))

    # 2. 遍历结束后，再统一执行替换
    for parent, child_name, old_module in replace_list:
        # 再次确认，防止对同一个实例重复操作
        if not isinstance(getattr(parent, child_name), AdaptiveConv2d):
            new_layer = AdaptiveConv2d(old_module)
            setattr(parent, child_name, new_layer)

    print(f"✅ 成功包装了 {len(replace_list)} 个卷积层，未触发递归。")
    # # ===========================================================

    # ========== [ 7. 最终同步与保存 ] ==========
    if target_attn: target_attn.forward = original_attn_forward

    model = force_bn_stats_align(model)
    model = final_physical_align(model)

    def count_parameters(m):
        return sum(p.numel() for p in m.parameters())

    final_params = count_parameters(model)
    print(f"\n📊 剪枝后物理参数量: {final_params / 1e6:.2f} M")

    yolo_container.model = model.to(target_device)
    if 'report_detailed_physical_channels' in globals():
        report_detailed_physical_channels(model, title="Final Physical Report")

    print("✅ 剪枝全流程成功完成\n")
    return yolo_container


import ultralytics.nn.tasks as tasks


def disable_fuse(self):
    """强制返回模型本身，不进行任何融合操作"""
    return self


def prune(args):
    # 0. 日志过滤
    warnings.filterwarnings('ignore', category=UserWarning)
    ultralytics_logger = logging.getLogger("ultralytics")
    log_filter = PruningLogFilter()
    ultralytics_logger.addFilter(log_filter)

    # 1. 初始化 YOLO 容器
    print(f"🚀 正在初始化 YOLO11 模型: {args.model}")
    yolo = YOLO(args.model)
    tasks.DetectionModel.fuse = disable_fuse
    pruning_cfg = {
        'data': "/gemini/code/Torch-Pruning-master/examples/yolov8/data.yaml",
        'imgsz': 160,
        'epochs': args.epochs if hasattr(args, 'epochs') else 500,
        'workers': 16,
        'batch': 32,
        'lr0' : 0.01,
        'project': '/gemini/output'
    }

    target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo.model.to(target_device).train()

    # 3. 预计算基准 (针对内部模型)
    example_inputs = torch.randn(1, 3, 640, 640).to(target_device)
    # 归一化输入以消除 YOLO 警告
    example_inputs = torch.clamp(example_inputs, 0, 1)

    base_macs, base_params = tp.utils.count_ops_and_params(yolo.model, example_inputs)
    print(f"📊 Baseline - Params: {base_params / 1e6:.2f}M, MACs: {base_macs / 1e9:.2f}G")

    # 4. 迭代参数
    total_iterations = 1
    step_ratio = args.target_prune_rate / total_iterations

    # 5. 迭代剪枝
    for i in range(total_iterations):
        print(f"\n" + "=" * 60)
        print(f">>>>> 第 {i + 1} / {total_iterations} 轮迭代循环开始 <<<<<")

        # A. 执行安全剪枝 (传入容器，返回修改后的容器)
        yolo = safe_prune_model(
            yolo_container=yolo,
            pruning_ratio=step_ratio,
            example_inputs=example_inputs,
            target_device=target_device
        )
        # B. 统计当前指标 (必须针对 yolo.model，避免 ZeroDivisionError)
        yolo.model.eval()
        with torch.no_grad():
            try:
                # 重新准备干净的输入进行 trace
                trace_input = torch.randn(1, 3, 640, 640).to(target_device).clamp(0, 1)

                macs, params = tp.utils.count_ops_and_params(yolo.model, trace_input)
                print(f"📉 剪枝后 - Params: {params / 1e6:.3f} M, MACs: {macs / 1e9:.3f} G")
            except ZeroDivisionError:
                params = sum(p.numel() for p in yolo.model.parameters())
                print(f"📉 剪枝后 - Params: {params / 1e6:.3f} M (MACs 统计跳过)")
        yolo.model.train()

        # C. 精度恢复微调
        # 重新绑定自定义训练函数到容器
        yolo.train_v2 = train_v2.__get__(yolo)
        pruning_cfg['name'] = f"iter_prune_step_{i + 1}"

        print(f"🔄 启动第 {i + 1} 轮微调...")
        yolo.train_v2(pruning=True, **pruning_cfg)
        yolo.model.fuse = disable_fuse.__get__(yolo.model, tasks.DetectionModel)
        # D. 验证与保存
        # metrics = yolo.val(data=pruning_cfg['data'], imgsz=640, batch=1, verbose=False)
        # print(f"✅ 本轮 mAP: {metrics.box.map:.4f}")

        save_path = Path(yolo.trainer.save_dir) / f"pruned_step_{i + 1}.pt"
        yolo.save(str(save_path))

    # 6. 最终导出
    print("\n" + "★" * 60)
    onnx_path = yolo.export(format='onnx', imgsz=160, simplify=True)
    print(f"✨ 任务完成！最终 ONNX 路径: {onnx_path}")
    print("★" * 60)

    return yolo

def predict(args):
    model = YOLO("/gemini/code/Torch-Pruning-master/examples/yolov8/test_model.pt")
    results = model.predict(
        source=r"/gemini/code/Torch-Pruning-master/examples/yolov8/test_image/test1.jpg",
        data=r"/gemini/code/Torch-Pruning-master/examples/yolov8/data.yaml", 
        save=True,
        show=True
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # YOLO11模型路径
    parser.add_argument('--model', default='/gemini/code/Torch-Pruning-master/examples/yolov8/yolo11n.pt',
                        help='Pretrained pruning target model file (YOLO11)')
    parser.add_argument('--cfg', default='default.yaml',
                        help='Pruning config file (same format as ultralytics default.yaml)')
    parser.add_argument('--target-prune-rate', default=0, type=float, help='剪枝率')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')

    args = parser.parse_args()
    prune(args)
    # predict(args)