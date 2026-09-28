from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

SCRIPT_VARIANT = "main_lg_branch_weights"

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Speed on Ampere/Ada GPUs. This does not change the model definition.
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_builder import DATASET_PRESETS, build_scene_dataloaders  # noqa: E402
from models.sadrfnet_lg import StageDRBConfig, sadrfnet50_lg  # noqa: E402


# =============================================================================
# USER CONFIG: modify only this area for normal experiments
# =============================================================================
MODEL_VARIANT = "sadrfnet_lg"

# Running mode:
#   "train"      -> train only
#   "test"       -> test only, using CHECKPOINT_PATH
#   "train_test" -> train first, then automatically test best.pth
MODE = "train_test"

# Dataset switch:
#   "AID"              commonly uses TRAIN_RATIO = 0.20 or 0.50
#   "NWPU-RESISC45"    commonly uses TRAIN_RATIO = 0.10 or 0.20
#   "UCMerced_LandUse" commonly uses TRAIN_RATIO = 0.50 or 0.80
DATASET_NAME = "UCMerced_LandUse"
TRAIN_RATIO = 0.80
# 是否从训练集中再划分验证集。
# 0.0 表示不划分验证集，训练过程中直接用测试集评估并保存 best.pth。
# 0.1 表示从训练集里拿出 10% 作为验证集，测试集只用于最终测试，更符合严格论文实验。
# 小数据集常用 0.0；严格实验建议设为 0.1。
VAL_RATIO_WITHIN_TRAIN = 0.0
# 是否重新生成数据集划分文件。
# False：如果 splits 目录下已有 train/test 划分文件，就直接读取旧划分。
# True：强制重新划分数据集，会覆盖原有划分文件。
# 修改 TRAIN_RATIO、VAL_RATIO_WITHIN_TRAIN 或 seed 后，第一次运行建议设为 True。
OVERWRITE_SPLIT = False
# 输入图像尺寸。
# ResNet/ImageNet 预训练模型通常使用 224×224。
# 遥感场景分类中也常用 224。
IMAGE_SIZE = 224

EPOCHS = 120
NUM_RUNS = 1
BASE_SEED = 42  # 基础随机种子
SEED_STEP = 1   # 每次重复实验随机种子的递增步长;SEED_STEP = 1 表示 seed 依次为 42、43、44。如果设为 10，则为 42、52、62。

# 是否启用更严格的确定性计算。
# False：训练更快，但不同运行之间可能有轻微波动。
# True：结果更容易复现，但训练速度会变慢。如果要检查同一 seed 是否能复现，建议设为 True，并把 NUM_WORKERS 设为 0。
DETERMINISTIC = True

# 每个 batch 的图像数量。
# BATCH_SIZE 越大，训练越快，但显存占用越大。显存不足时可以改为 16 或 8。
BATCH_SIZE = 32
# DataLoader 读取数据的子进程数量。
# NUM_WORKERS = 0：主进程读取数据，最容易复现，但较慢。# NUM_WORKERS = 4：常用设置，读取速度较快。# NUM_WORKERS = 8：数据集较大、CPU 资源足够时可尝试。
NUM_WORKERS = 4

# 每隔多少个 epoch 做一次验证/测试评估。
# VALIDATE_EVERY = 1：每个 epoch 都评估，best.pth 选择最精确，但较慢。
# VALIDATE_EVERY = 5：每 5 个 epoch 评估一次，训练更快。
# VALIDATE_EVERY = 10：适合大数据集快速筛选。正式论文实验建议设为 1。
VALIDATE_EVERY = 5
# 是否关闭 tqdm 进度条。
# True：终端输出更简洁，适合远程服务器。
# False：显示训练和测试进度条。
DISABLE_TQDM = True

# Model:
# 渐进式融合分支中的通道数。
# 256 表示将 x1、x2、x3、x4 都投影到 256 通道后进行渐进式融合。
# 数值越大，融合分支表达能力越强，但计算量和显存也增加。
FUSION_CHANNELS = 256
# 最终分类前的特征维度。
# 2048 与 ResNet-50 stage4 的输出通道数一致，有利于保留 ImageNet 预训练语义。
# 如果设为 512，会更轻量，但可能削弱预训练高层语义。
EMBED_DIM = 2048
# 最终融合模块和 MLP 中的 Dropout 比例。
DROPOUT = 0.2 #0.2

# DRB 使用方式。
# "adapter"：新版推荐模式。完整 ResNet stage 作为局部路径，
#             独立 DRB context path 作为全局路径，再进行局部—全局动态融合。
# "replace"：消融模式。直接用 DRB bottleneck 替换部分 ResNet bottleneck。
DRB_MODE = "adapter"

# 每个 stage 的 DRB 全局路径深度比例。
# 新版 adapter 模式下不会替换原始 ResNet block，而是额外建立 DRB 全局路径：
# Stage1: 3×0.00 = 0 个全局单元，保留低层边缘/纹理；
# Stage2: 4×0.25 = 1 个 5×5 DRB 单元，建模中尺度地物结构；
# Stage3: 6×0.33 ≈ 2 个 7×7 DRB 单元，建模大尺度场景上下文；
# Stage4: 3×0.00 = 0 个全局单元，保留稳定高层语义锚点。
DRB_STAGE_RATIOS = (0.0, 0.25, 0.33, 0.0)

# 各 stage 的大核尺寸。只有 ratio > 0 的 stage 实际启用。
DRB_LARGE_KERNELS = (3, 5, 9, 5)

# 各 stage 的多尺度空洞率。
DRB_DILATIONS = ((1,), (1, 2), (1, 2, 3), (1, 2))

# DRB context unit 内 ConvFFN 的扩展比例。
DRB_FFN_EXPANSIONS = (1.0, 1.0, 1.0, 1.0)

# 仅为兼容旧配置保留。新版 local-global adapter 模式不使用
# sigmoid(adapter_init) 弱辅助门控；replace 消融也不依赖该值。
DRB_ADAPTER_INIT = -4.0

# 是否在 DRB 全局上下文单元中使用 SE + ConvFFN。
DRB_USE_SE_FFN = True

# 全局路径的初始残差尺度：
# global = local + tanh(scale) * DRB_context。
# 0.05 使随机初始化的全局路径在训练初期接近预训练局部路径，
# 但局部与全局在后续融合中仍具有独立、对等的路由权重。
DRB_GLOBAL_RESIDUAL_INIT = 0.05

# 局部—全局通道路由器的初始局部先验。
# 0.50 表示初始 Local/Global 权重均为 0.50。
DRB_FUSION_LOCAL_PRIOR = 0.50

# 是否使用 ImageNet 预训练权重。
# True：推荐。将 ResNet-50 的 ImageNet 预训练参数加载到可匹配的层中。
# False：从零训练，不建议用于 UC Merced / AID / NWPU 这类数据集。
USE_IMAGENET_PRETRAINED = True
# 本地 ImageNet ResNet-50 权重路径。
# 如果留空，torchvision 会自动下载或读取缓存中的 ResNet-50 权重。
# 如果服务器不能联网，可以手动填写本地路径。
# 示例：
# IMAGENET_PRETRAINED_PATH = "/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
IMAGENET_PRETRAINED_PATH = ""
# 是否打印 ImageNet 预训练权重加载信息。
# True：显示加载了多少层、哪些模块是新初始化的。False：不打印。
IMAGENET_PRETRAINED_VERBOSE = True

# 优化器类型。
# "adamw"：推荐，训练稳定，适合当前模型。
# "sgd"：传统 CNN 常用，但需要更细致调学习率和动量。
OPTIMIZER_NAME = "adamw"
LR = 2e-4     # 基础学习率 1e-4
MIN_LR = 1e-6 # cosine 学习率调度的最低学习率。
WEIGHT_DECAY = 1e-4
# 预训练 backbone 的学习率倍率。
# 0.1 表示 backbone 实际学习率为 LR × 0.1 = 3e-5。
# 这样可以避免破坏 ImageNet 预训练语义。
BACKBONE_LR_MULT = 0.1
# 新增模块的学习率倍率。
# 1.0 表示 DRB 全局路径、局部—全局融合、渐进式融合、最终融合和分类器使用 LR × 1.0 = 3e-4。
# 新模块是随机初始化，需要更大的学习率来学习。
NEW_MODULE_LR_MULT = 1.0
# 是否对 BN、LayerNorm、bias 等参数关闭 weight decay。
# True：推荐。Norm 和 bias 通常不做权重衰减，训练更稳。
NO_WEIGHT_DECAY_ON_NORM_BIAS = True
# SGD 优化器的 momentum 参数。
# 只有 OPTIMIZER_NAME = "sgd" 时生效。
# AdamW 下基本不使用。
MOMENTUM = 0.9
# 学习率调度器类型。
# "cosine"：推荐，学习率平滑下降。
# "step"：每隔 STEP_SIZE 个 epoch 按 GAMMA 衰减。
# "none"：不使用学习率调度。
SCHEDULER_NAME = "cosine"
STEP_SIZE = 30 # StepLR 的步长。只有 SCHEDULER_NAME = "step" 时生效。
GAMMA = 0.1
LABEL_SMOOTHING = 0.01
GRAD_CLIP = 0.0
AMP = True

# Runtime:
# DEVICE: "auto", "cuda", or "cpu"
DEVICE = "auto"
SAVE_EVERY = 0 # 是否每隔 N 个 epoch 额外保存一个 checkpoint。0 表示不额外保存。

# 是否保存 last.pth。
# False：不保存每轮最后模型，减少磁盘占用和 torch.save 报错概率。
# True：保存 last.pth，方便断点续训，但会增加 I/O 和磁盘占用。
SAVE_LAST_CHECKPOINT = False
# 是否在 checkpoint 中保存 optimizer 和 scheduler 状态。
# False：只保存模型权重，文件小，适合普通实验和论文结果。
# True：保存完整训练状态，支持严格断点续训，但文件很大。
SAVE_OPTIMIZER_STATE = False
# 是否使用原子保存 checkpoint。
# True：先保存到临时文件，再替换为正式文件，降低 checkpoint 写坏风险。
# False：直接 torch.save 到目标文件
ATOMIC_CHECKPOINT_SAVE = True

# Checkpoints:
# RESUME_PATH is used only for continuing training.
# CHECKPOINT_PATH is used only when MODE = "test".
RESUME_PATH = "" # 断点续训路径。
CHECKPOINT_PATH = ""  # 独立测试时要加载的 checkpoint 路径。只有 MODE = "test" 时使用。
ALLOW_PARTIAL_LOAD = False # 是否允许部分加载 checkpoint。
SWITCH_TO_DEPLOY = False # 是否在测试前把 DRB 多分支结构切换为推理部署结构。
# 测试时增强。
# True：对原图、水平翻转、垂直翻转、水平+垂直翻转的 logits 求平均。
# 遥感图像方向通常不固定，因此 TTA 往往能提高一点精度。
# False：只用原图测试，速度更快。
TEST_TTA = True

# Output:
# Leave empty to use outputs/<dataset>_sadrfnet_lg_trainXX_seedXX/.
OUTPUT_DIR = ""
TEST_OUTPUT_DIR = ""


def build_args_from_user_config() -> argparse.Namespace:
    """Convert the editable USER CONFIG variables into the internal args object."""
    if DATASET_NAME not in DATASET_PRESETS:
        raise ValueError(f"Unsupported DATASET_NAME={DATASET_NAME}. Supported: {list(DATASET_PRESETS.keys())}")
    if MODE not in {"train", "test", "train_test"}:
        raise ValueError('MODE must be one of: "train", "test", "train_test".')
    if OPTIMIZER_NAME.lower() not in {"adamw", "sgd"}:
        raise ValueError('OPTIMIZER_NAME must be "adamw" or "sgd".')
    if SCHEDULER_NAME.lower() not in {"cosine", "step", "none"}:
        raise ValueError('SCHEDULER_NAME must be "cosine", "step", or "none".')
    if DEVICE not in {"auto", "cuda", "cpu"}:
        raise ValueError('DEVICE must be "auto", "cuda", or "cpu".')
    if NUM_RUNS < 1:
        raise ValueError("NUM_RUNS must be >= 1.")
    if EPOCHS < 1 and MODE in {"train", "train_test"}:
        raise ValueError("EPOCHS must be >= 1 for training.")

    return argparse.Namespace(
        model_variant=MODEL_VARIANT,
        mode=MODE,
        dataset=DATASET_NAME,
        train_ratio=TRAIN_RATIO,
        val_ratio_within_train=VAL_RATIO_WITHIN_TRAIN,
        overwrite_split=OVERWRITE_SPLIT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        validate_every=VALIDATE_EVERY,
        disable_tqdm=DISABLE_TQDM,
        fusion_channels=FUSION_CHANNELS,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
        drb_mode=DRB_MODE,
        drb_stage_ratios=DRB_STAGE_RATIOS,
        drb_large_kernels=DRB_LARGE_KERNELS,
        drb_dilations=DRB_DILATIONS,
        drb_ffn_expansions=DRB_FFN_EXPANSIONS,
        drb_adapter_init=DRB_ADAPTER_INIT,
        drb_use_se_ffn=DRB_USE_SE_FFN,
        drb_global_residual_init=DRB_GLOBAL_RESIDUAL_INIT,
        drb_fusion_local_prior=DRB_FUSION_LOCAL_PRIOR,
        use_imagenet_pretrained=USE_IMAGENET_PRETRAINED,
        imagenet_pretrained_path=IMAGENET_PRETRAINED_PATH,
        imagenet_pretrained_verbose=IMAGENET_PRETRAINED_VERBOSE,
        epochs=EPOCHS,
        optimizer=OPTIMIZER_NAME,
        lr=LR,
        min_lr=MIN_LR,
        weight_decay=WEIGHT_DECAY,
        backbone_lr_mult=BACKBONE_LR_MULT,
        new_module_lr_mult=NEW_MODULE_LR_MULT,
        no_weight_decay_on_norm_bias=NO_WEIGHT_DECAY_ON_NORM_BIAS,
        momentum=MOMENTUM,
        scheduler=SCHEDULER_NAME,
        step_size=STEP_SIZE,
        gamma=GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        grad_clip=GRAD_CLIP,
        amp=AMP,
        seed=BASE_SEED,
        num_runs=NUM_RUNS,
        seed_step=SEED_STEP,
        deterministic=DETERMINISTIC,
        device=DEVICE,
        resume=RESUME_PATH,
        checkpoint=CHECKPOINT_PATH,
        allow_partial_load=ALLOW_PARTIAL_LOAD,
        switch_to_deploy=SWITCH_TO_DEPLOY,
        test_tta=TEST_TTA,
        output_dir=OUTPUT_DIR,
        test_output_dir=TEST_OUTPUT_DIR,
        save_every=SAVE_EVERY,
        save_last_checkpoint=SAVE_LAST_CHECKPOINT,
        save_optimizer_state=SAVE_OPTIMIZER_STATE,
        atomic_checkpoint_save=ATOMIC_CHECKPOINT_SAVE,
    )


def print_user_config(args: argparse.Namespace) -> None:
    """Print the effective experiment configuration."""
    print("=" * 80)
    print("[USER CONFIG]")
    print(f"SCRIPT_VARIANT = {SCRIPT_VARIANT}")
    print(f"MODEL_VARIANT = {args.model_variant}")
    print(f"MODE = {args.mode}")
    print(f"DATASET_NAME = {args.dataset}")
    print(f"TRAIN_RATIO = {args.train_ratio}")
    print(f"VAL_RATIO_WITHIN_TRAIN = {args.val_ratio_within_train}")
    print(f"EPOCHS = {args.epochs}")
    print(f"NUM_RUNS = {args.num_runs}")
    print(f"BASE_SEED = {args.seed}")
    print(f"SEED_STEP = {args.seed_step}")
    print(f"DETERMINISTIC = {args.deterministic}")
    print(f"BATCH_SIZE = {args.batch_size}")
    print(f"NUM_WORKERS = {args.num_workers}")
    print(f"VALIDATE_EVERY = {args.validate_every}")
    print(f"DISABLE_TQDM = {args.disable_tqdm}")
    print(f"DRB_MODE = {args.drb_mode}")
    print(f"DRB_STAGE_RATIOS = {args.drb_stage_ratios}")
    print(f"DRB_ADAPTER_INIT = {args.drb_adapter_init} (legacy compatibility)")
    print(f"DRB_GLOBAL_RESIDUAL_INIT = {args.drb_global_residual_init}")
    print(f"DRB_FUSION_LOCAL_PRIOR = {args.drb_fusion_local_prior}")
    print(f"USE_IMAGENET_PRETRAINED = {args.use_imagenet_pretrained}")
    print(f"IMAGENET_PRETRAINED_PATH = {args.imagenet_pretrained_path}")
    print(f"OPTIMIZER_NAME = {args.optimizer}")
    print(f"LR = {args.lr}")
    print(f"WEIGHT_DECAY = {args.weight_decay}")
    print(f"BACKBONE_LR_MULT = {args.backbone_lr_mult}")
    print(f"NEW_MODULE_LR_MULT = {args.new_module_lr_mult}")
    print(f"SCHEDULER_NAME = {args.scheduler}")
    print(f"AMP = {args.amp}")
    print(f"TEST_TTA = {args.test_tta}")
    print(f"SAVE_LAST_CHECKPOINT = {args.save_last_checkpoint}")
    print(f"SAVE_OPTIMIZER_STATE = {args.save_optimizer_state}")
    print("=" * 80)



# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Usually faster for fixed-size images such as 224x224 remote-sensing scenes.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device: {device_arg}")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


class MetricTracker:
    """Collect predictions and compute OA, AA, Kappa and macro scores."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.preds: List[int] = []
        self.targets: List[int] = []
        self.confidences: List[float] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = torch.softmax(logits.detach(), dim=1)
        conf, preds = torch.max(probs, dim=1)
        self.preds.extend(preds.cpu().numpy().tolist())
        self.targets.extend(labels.detach().cpu().numpy().tolist())
        self.confidences.extend(conf.cpu().numpy().tolist())

    def compute(self) -> Dict[str, object]:
        metrics = compute_metrics(self.targets, self.preds, self.num_classes)
        metrics["mean_confidence"] = float(np.mean(self.confidences)) if self.confidences else 0.0
        return metrics


def build_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(targets, preds):
        t = int(target)
        p = int(pred)
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_metrics(targets: Iterable[int], preds: Iterable[int], num_classes: int) -> Dict[str, object]:
    targets = np.asarray(list(targets), dtype=np.int64)
    preds = np.asarray(list(preds), dtype=np.int64)
    cm = build_confusion_matrix(targets, preds, num_classes)

    total = int(cm.sum())
    correct = int(np.trace(cm))
    oa = float(correct / total) if total > 0 else 0.0

    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = np.divide(
        per_class_correct,
        np.maximum(per_class_total, 1),
        out=np.zeros_like(per_class_correct, dtype=np.float64),
        where=per_class_total > 0,
    )
    aa = float(per_class_acc.mean()) if num_classes > 0 else 0.0

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    precision = np.divide(tp, np.maximum(tp + fp, 1.0))
    recall = np.divide(tp, np.maximum(tp + fn, 1.0))
    f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1e-12))

    row_marginal = cm.sum(axis=1)
    col_marginal = cm.sum(axis=0)
    expected = float(np.dot(row_marginal, col_marginal) / max(total * total, 1))
    kappa = float((oa - expected) / max(1.0 - expected, 1e-12))

    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "macro_precision": float(precision.mean()) if num_classes > 0 else 0.0,
        "macro_recall": float(recall.mean()) if num_classes > 0 else 0.0,
        "macro_f1": float(f1.mean()) if num_classes > 0 else 0.0,
        "per_class_acc": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
    }


# -----------------------------------------------------------------------------
# Model, optimizer, scheduler, checkpoint
# -----------------------------------------------------------------------------


def build_stage_cfgs(args: argparse.Namespace):
    """Build four StageDRBConfig objects from the editable config area."""
    if len(args.drb_stage_ratios) != 4:
        raise ValueError("DRB_STAGE_RATIOS must contain four values.")
    if len(args.drb_large_kernels) != 4:
        raise ValueError("DRB_LARGE_KERNELS must contain four values.")
    if len(args.drb_dilations) != 4:
        raise ValueError("DRB_DILATIONS must contain four tuples.")
    if len(args.drb_ffn_expansions) != 4:
        raise ValueError("DRB_FFN_EXPANSIONS must contain four values.")

    cfgs = []
    for ratio, kernel, dilations, ffn_expansion in zip(
        args.drb_stage_ratios,
        args.drb_large_kernels,
        args.drb_dilations,
        args.drb_ffn_expansions,
    ):
        cfgs.append(
            StageDRBConfig(
                replace_ratio=float(ratio),
                large_kernel_size=int(kernel),
                dilations=tuple(int(d) for d in dilations),
                ffn_expansion=float(ffn_expansion),
                mode=str(args.drb_mode),
                adapter_init=float(args.drb_adapter_init),
                use_se_ffn=bool(args.drb_use_se_ffn),
                global_residual_init=float(args.drb_global_residual_init),
                fusion_local_prior=float(args.drb_fusion_local_prior),
            )
        )
    return tuple(cfgs)


def build_model(
    num_classes: int,
    args: argparse.Namespace,
    load_imagenet_pretrained: Optional[bool] = None,
) -> nn.Module:
    """Build SADRFNet-LG and optionally initialize matched layers from ImageNet ResNet-50."""
    if load_imagenet_pretrained is None:
        load_imagenet_pretrained = bool(args.use_imagenet_pretrained)
    model = sadrfnet50_lg(
        num_classes=num_classes,
        in_chans=3,
        fusion_channels=args.fusion_channels,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        return_features=False,
        deploy=False,
        stage_cfgs=build_stage_cfgs(args),
        pretrained=bool(load_imagenet_pretrained),
        pretrained_path=args.imagenet_pretrained_path or None,
        pretrained_verbose=bool(args.imagenet_pretrained_verbose),
    )
    model.model_variant = args.model_variant
    return model


def _is_new_or_random_module(name: str) -> bool:
    """Parameters that are newly introduced or heavily changed from ResNet-50."""
    keywords = (
        "progressive_fusion",
        "final_fusion",
        "classifier",
        "spatial",
        "drb",
        "adapter_logit",
        "post",
    )
    return any(k in name for k in keywords)


def _is_norm_or_bias(name: str, param: torch.nn.Parameter) -> bool:
    lname = name.lower()
    return (
        param.ndim <= 1
        or name.endswith(".bias")
        or "bn" in lname
        or "norm" in lname
        or "layernorm" in lname
    )


def build_optimizer(model: nn.Module, args: argparse.Namespace):
    """
    Build optimizer with parameter groups.

    Rationale:
    - ImageNet-pretrained stem/vanilla bottleneck parameters should be fine-tuned
      with a smaller LR.
    - Newly introduced DRB global paths, local-global routers, progressive fusion, final fusion,
      and classifier need a larger LR to learn from scratch.
    - Norm parameters and biases usually should not receive weight decay.
    """
    name = args.optimizer.lower()
    param_groups: Dict[Tuple[float, float], Dict[str, object]] = {}

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_new = _is_new_or_random_module(param_name)
        lr = args.lr * (args.new_module_lr_mult if is_new else args.backbone_lr_mult)
        if bool(args.no_weight_decay_on_norm_bias) and _is_norm_or_bias(param_name, param):
            wd = 0.0
        else:
            wd = args.weight_decay

        key = (float(lr), float(wd))
        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "lr": float(lr),
                "weight_decay": float(wd),
                "group_name": f"lr{lr:.2e}_wd{wd:.2e}",
            }
        param_groups[key]["params"].append(param)

    groups = list(param_groups.values())
    print("[Optimizer] Parameter groups:")
    for group in groups:
        n = sum(p.numel() for p in group["params"])
        print(f"  {group['group_name']}: params={n:,}")

    if name == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=args.momentum, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")



def build_scheduler(optimizer, args: argparse.Namespace):
    name = args.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_oa: float,
    args: argparse.Namespace,
    metrics: Dict[str, object],
) -> None:
    """
    Save checkpoints safely.

    By default this file saves model weights only. This avoids very large
    optimizer-state files and reduces the chance of torch.save zip errors on
    network/cloud disks during long multi-run experiments.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_optimizer = bool(getattr(args, "save_optimizer_state", False))
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if (save_optimizer and optimizer is not None) else None,
        "scheduler": scheduler.state_dict() if (save_optimizer and scheduler is not None) else None,
        "best_oa": float(best_oa),
        "best_acc": float(best_oa),  # backward-compatible name
        "args": vars(args),
        "metrics": metrics,
    }

    if bool(getattr(args, "atomic_checkpoint_save", True)):
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    else:
        torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    strict: bool = True,
) -> Tuple[int, float, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict):
        checkpoint_args = checkpoint.get("args", {})
        if isinstance(checkpoint_args, dict):
            checkpoint_variant = checkpoint_args.get("model_variant")
            expected_variant = getattr(model, "model_variant", "sadrfnet_lg")
            if checkpoint_variant is not None and checkpoint_variant != expected_variant:
                raise RuntimeError(
                    f"Checkpoint model_variant={checkpoint_variant!r} does not match "
                    f"the current model_variant={expected_variant!r}. "
                    "Use a checkpoint produced by main_lg.py."
                )

    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state_dict = strip_module_prefix(state_dict)

    if strict:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    start_epoch = 1
    best_oa = 0.0
    if isinstance(checkpoint, dict):
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_oa = float(checkpoint.get("best_oa", checkpoint.get("best_acc", 0.0)))
        if optimizer is not None and checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

    info = {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint_epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "checkpoint_best_oa": best_oa,
    }
    return start_epoch, best_oa, info


# -----------------------------------------------------------------------------
# Train and evaluate loops
# -----------------------------------------------------------------------------


def get_logits(outputs):
    return outputs["logits"] if isinstance(outputs, dict) else outputs


@torch.no_grad()
def forward_with_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Average logits from original, horizontal, vertical and hv-flipped inputs."""
    logits = get_logits(model(images))
    logits = logits + get_logits(model(torch.flip(images, dims=[3])))
    logits = logits + get_logits(model(torch.flip(images, dims=[2])))
    logits = logits + get_logits(model(torch.flip(images, dims=[2, 3])))
    return logits / 4.0


def progress(iterable, desc: str):
    if tqdm is None or DISABLE_TQDM:
        return iterable
    return tqdm(iterable, desc=desc, ncols=100)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in progress(loader, f"Train Epoch {epoch}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        use_amp = bool(args.amp and device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = get_logits(model(images))
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == labels).sum().item())
        total += batch_size

    return {
        "loss": loss_meter.avg,
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    num_classes: int,
    desc: str = "Eval",
    tta: bool = False,
) -> Dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    tracker = MetricTracker(num_classes)

    for images, labels in progress(loader, desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        if tta:
            logits = forward_with_tta(model, images)
        else:
            logits = get_logits(model(images))
        loss = criterion(logits, labels)

        loss_meter.update(float(loss.item()), labels.size(0))
        tracker.update(logits, labels)

    metrics = tracker.compute()
    metrics["loss"] = loss_meter.avg
    return metrics


# -----------------------------------------------------------------------------
# Logging and output files
# -----------------------------------------------------------------------------


def append_csv_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_test_outputs(output_dir: Path, metrics: Dict[str, object], class_names: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in metrics.items() if k not in {"confusion_matrix", "per_class_acc"}}
    save_json(output_dir / "test_metrics.json", summary)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for idx, row in enumerate(cm):
            writer.writerow([class_names[idx]] + row.tolist())

    per_class_acc = metrics["per_class_acc"]
    with (output_dir / "per_class_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "accuracy"])
        for idx, acc in enumerate(per_class_acc):
            writer.writerow([idx, class_names[idx], acc])


def print_metrics(prefix: str, metrics: Dict[str, object]) -> None:
    print(
        f"[{prefix}] "
        f"loss={float(metrics.get('loss', 0.0)):.6f} "
        f"OA={float(metrics['oa']):.6f} "
        f"AA={float(metrics['aa']):.6f} "
        f"Kappa={float(metrics['kappa']):.6f} "
        f"Macro-F1={float(metrics.get('macro_f1', 0.0)):.6f}"
    )



def _to_python_value(value):
    """Recursively convert tensors/NumPy values to plain Python values."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_python_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python_value(v) for v in value]
    return value


def get_local_global_weights_for_output(model: nn.Module) -> Dict[str, object]:
    """
    Directly call model.get_local_global_branch_weights().

    The model stores routing weights from its latest forward pass, so these
    values normally correspond to the final test batch processed by evaluate().
    """
    if not hasattr(model, "get_local_global_branch_weights"):
        return {
            "available": False,
            "message": (
                "The current model does not implement "
                "get_local_global_branch_weights()."
            ),
        }

    try:
        weights = model.get_local_global_branch_weights()
    except Exception as exc:
        return {
            "available": False,
            "message": f"Failed to read branch weights: {exc}",
        }

    return {
        "available": True,
        "weights": _to_python_value(weights),
    }

def get_effective_train_ratio(args: argparse.Namespace) -> float:
    if args.train_ratio is not None:
        return float(args.train_ratio)
    return float(DATASET_PRESETS[args.dataset].default_train_ratio)


def make_output_dir(args: argparse.Namespace, run_seed: int, run_index: int = 0) -> Path:
    train_ratio = get_effective_train_ratio(args)
    if args.output_dir:
        base = Path(args.output_dir)
        if args.num_runs > 1:
            return base / f"run{run_index + 1:02d}_seed{run_seed}"
        return base

    suffix = f"{args.dataset}_sadrfnet_lg_train{int(round(train_ratio * 100)):02d}_seed{run_seed}"
    if args.num_runs > 1:
        suffix = f"{suffix}_run{run_index + 1:02d}"
    return PROJECT_ROOT / "outputs" / suffix


# -----------------------------------------------------------------------------
# Main routines
# -----------------------------------------------------------------------------


def build_data(args: argparse.Namespace, seed: int):
    return build_scene_dataloaders(
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite_split=args.overwrite_split,
        project_root=PROJECT_ROOT,
    )


def run_train(args: argparse.Namespace, run_seed: Optional[int] = None, run_index: int = 0) -> Dict[str, object]:
    run_seed = int(args.seed if run_seed is None else run_seed)
    set_seed(run_seed, deterministic=args.deterministic)

    device = get_device(args.device)
    output_dir = make_output_dir(args, run_seed, run_index)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_args = copy.deepcopy(vars(args))
    run_args["model_variant"] = args.model_variant
    run_args["effective_seed"] = run_seed
    run_args["effective_train_ratio"] = get_effective_train_ratio(args)
    save_json(output_dir / "main_args.json", run_args)

    print(f"[Runtime] Mode: train")
    print(f"[Runtime] Device: {device}")
    print(f"[Runtime] Seed: {run_seed}")
    print(f"[Runtime] Output dir: {output_dir}")

    data = build_data(args, seed=run_seed)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_names = data["class_names"]
    num_classes = int(data["num_classes"])

    print(f"[Data] Dataset: {data['dataset_name']}")
    print(f"[Data] Dataset root: {data['dataset_root']}")
    print(f"[Data] Split dir: {data['split_dir']}")
    print(f"[Data] Classes: {num_classes}")
    print(
        f"[Data] Train/Val/Test samples: "
        f"{len(data['train_dataset'])}/"
        f"{0 if data['val_dataset'] is None else len(data['val_dataset'])}/"
        f"{len(data['test_dataset'])}"
    )

    if val_loader is None:
        eval_loader = test_loader
        eval_name = "test"
        print("[Warning] No validation split was created.")
        print("[Warning] Best checkpoint will be selected on the test set. For stricter papers, use --val-ratio-within-train 0.1.")
    else:
        eval_loader = val_loader
        eval_name = "val"

    # Load ImageNet initialization only for training from scratch. If RESUME_PATH
    # is set, the checkpoint will fully define the model state instead.
    load_pretrained = bool(args.use_imagenet_pretrained and not args.resume)
    model = build_model(num_classes, args, load_imagenet_pretrained=load_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_oa = 0.0
    if args.resume:
        start_epoch, best_oa, info = load_checkpoint(
            Path(args.resume),
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            strict=not args.allow_partial_load,
        )
        print(f"[Resume] Loaded: {args.resume}")
        print(f"[Resume] Start epoch: {start_epoch}, best OA: {best_oa:.6f}")
        if info["missing_keys"] or info["unexpected_keys"]:
            print(f"[Resume] Missing keys: {info['missing_keys']}")
            print(f"[Resume] Unexpected keys: {info['unexpected_keys']}")

    best_metrics: Dict[str, object] = {}
    last_eval_stats: Optional[Dict[str, object]] = None
    validate_every = max(int(args.validate_every), 1)

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, args)

        do_eval = (epoch == 1) or (epoch == args.epochs) or (epoch % validate_every == 0)
        if do_eval:
            eval_stats = evaluate(model, eval_loader, criterion, device, num_classes, desc=f"{eval_name.capitalize()} Epoch {epoch}")
            last_eval_stats = eval_stats
        else:
            # Skip validation/test evaluation to save time. The best checkpoint is only updated on evaluated epochs.
            eval_stats = {
                "loss": float("nan"),
                "oa": float("nan"),
                "aa": float("nan"),
                "kappa": float("nan"),
                "macro_f1": float("nan"),
                "confusion_matrix": [],
                "per_class_acc": [],
            }

        if scheduler is not None:
            scheduler.step()
        lr = float(optimizer.param_groups[0]["lr"])

        is_best = False
        if do_eval:
            eval_oa = float(eval_stats["oa"])
            is_best = eval_oa > best_oa
            if is_best:
                best_oa = eval_oa
                best_metrics = eval_stats

        row = {
            "epoch": epoch,
            "seed": run_seed,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "evaluated": int(do_eval),
            f"{eval_name}_loss": eval_stats["loss"],
            f"{eval_name}_oa": eval_stats["oa"],
            f"{eval_name}_aa": eval_stats["aa"],
            f"{eval_name}_kappa": eval_stats["kappa"],
            f"{eval_name}_macro_f1": eval_stats["macro_f1"],
            "best_oa": best_oa,
        }
        append_csv_row(output_dir / "train_log.csv", row)

        if do_eval:
            print(
                f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                f"lr={lr:.6g} "
                f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
                f"{eval_name}_loss={eval_stats['loss']:.4f} "
                f"{eval_name}_OA={eval_stats['oa']:.4f} "
                f"{eval_name}_AA={eval_stats['aa']:.4f} "
                f"{eval_name}_Kappa={eval_stats['kappa']:.4f} "
                f"best_OA={best_oa:.4f}"
            )
        else:
            print(
                f"[Epoch {epoch:03d}/{args.epochs:03d}] "
                f"lr={lr:.6g} "
                f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
                f"{eval_name}=skipped(validate_every={validate_every}) "
                f"best_OA={best_oa:.4f}"
            )

        metrics_for_ckpt = last_eval_stats if last_eval_stats is not None else eval_stats
        if bool(getattr(args, "save_last_checkpoint", False)):
            save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, best_oa, args, metrics_for_ckpt)

        if is_best:
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, epoch, best_oa, args, eval_stats)
            save_json(output_dir / "best_metrics.json", {k: v for k, v in eval_stats.items() if k not in {"confusion_matrix", "per_class_acc"}})
            print(f"[Checkpoint] Saved best checkpoint: {output_dir / 'best.pth'}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, best_oa, args, metrics_for_ckpt)

    print(f"[Done] Best {eval_name} OA: {best_oa:.6f}")
    print(f"[Done] Checkpoints saved under: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / "best.pth"),
        "last_checkpoint": str(output_dir / "last.pth") if bool(getattr(args, "save_last_checkpoint", False)) else "",
        "best_oa": best_oa,
        "best_metrics": best_metrics,
        "eval_name": eval_name,
        "seed": run_seed,
        "class_names": class_names,
    }


def run_test(args: argparse.Namespace, checkpoint: Optional[str] = None, run_seed: Optional[int] = None) -> Dict[str, object]:
    seed = int(args.seed if run_seed is None else run_seed)
    set_seed(seed, deterministic=args.deterministic)
    device = get_device(args.device)

    ckpt_path = Path(checkpoint or args.checkpoint)
    if not ckpt_path.exists():
        # Convenient fallback: if checkpoint is not provided, try default output directory / best.pth.
        if not checkpoint and not args.checkpoint:
            default_output_dir = make_output_dir(args, seed, 0)
            fallback = default_output_dir / "best.pth"
            if fallback.exists():
                ckpt_path = fallback
            else:
                raise FileNotFoundError(
                    "Test mode requires --checkpoint. No default best.pth was found at: "
                    f"{fallback}"
                )
        else:
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    print(f"[Runtime] Mode: test")
    print(f"[Runtime] Device: {device}")
    print(f"[Runtime] Seed: {seed}")
    print(f"[Checkpoint] Loading: {ckpt_path}")

    # Do not overwrite splits during independent testing.
    old_overwrite = args.overwrite_split
    args.overwrite_split = False
    data = build_data(args, seed=seed)
    args.overwrite_split = old_overwrite

    test_loader = data["test_loader"]
    num_classes = int(data["num_classes"])
    class_names = data["class_names"]

    print(f"[Data] Dataset: {data['dataset_name']}")
    print(f"[Data] Dataset root: {data['dataset_root']}")
    print(f"[Data] Split dir: {data['split_dir']}")
    print(f"[Data] Classes: {num_classes}")
    print(f"[Data] Test samples: {len(data['test_dataset'])}")

    # In test mode, do not reload ImageNet weights before loading the trained checkpoint.
    model = build_model(num_classes, args, load_imagenet_pretrained=False).to(device)
    _, checkpoint_best_oa, info = load_checkpoint(
        ckpt_path,
        model,
        optimizer=None,
        scheduler=None,
        map_location=device,
        strict=not args.allow_partial_load,
    )
    print(f"[Checkpoint] checkpoint_best_oa: {checkpoint_best_oa:.6f}")
    if info["missing_keys"] or info["unexpected_keys"]:
        print(f"[Checkpoint] Missing keys: {info['missing_keys']}")
        print(f"[Checkpoint] Unexpected keys: {info['unexpected_keys']}")

    if args.switch_to_deploy:
        model.eval()
        model.switch_to_deploy()
        print("[Deploy] DRB branches have been fused by model.switch_to_deploy().")

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device, num_classes, desc="Test", tta=bool(args.test_tta))


    # Read branch weights from the model's latest forward pass.
    local_global_branch_weights = get_local_global_weights_for_output(model)

    if args.test_output_dir:
        output_dir = Path(args.test_output_dir)
    else:
        output_dir = ckpt_path.parent / "test_results"
    save_test_outputs(output_dir, metrics, class_names)

    print_metrics("Test Results", metrics)
    print(f"[Saved] Test outputs: {output_dir}")

    return {
        "checkpoint": str(ckpt_path),
        "output_dir": str(output_dir),
        "metrics": metrics,
        "seed": seed,
        "local_global_branch_weights": local_global_branch_weights,
    }


def run_multiple(args: argparse.Namespace) -> None:
    if args.num_runs < 1:
        raise ValueError("NUM_RUNS must be >= 1.")
    if args.num_runs > 1 and args.resume:
        raise ValueError("RESUME_PATH is only supported for a single run. Set NUM_RUNS = 1 when resuming.")
    if args.mode == "test" and args.num_runs > 1:
        raise ValueError("Multiple test runs require different checkpoints. Set NUM_RUNS = 1 for MODE = 'test'.")

    all_rows = []
    all_local_global_branch_weights: List[Dict[str, object]] = []
    for run_idx in range(args.num_runs):
        run_seed = int(args.seed + run_idx * args.seed_step)
        print("=" * 80)
        print(f"[Run] {run_idx + 1}/{args.num_runs}, seed={run_seed}")
        print("=" * 80)

        if args.mode == "train":
            result = run_train(args, run_seed=run_seed, run_index=run_idx)
            row = {
                "run": run_idx + 1,
                "seed": run_seed,
                "best_oa": result["best_oa"],
                "output_dir": result["output_dir"],
            }

        elif args.mode == "train_test":
            train_result = run_train(args, run_seed=run_seed, run_index=run_idx)
            test_result = run_test(args, checkpoint=train_result["best_checkpoint"], run_seed=run_seed)
            m = test_result["metrics"]
            all_local_global_branch_weights.append({
                "run": run_idx + 1,
                "seed": run_seed,
                "result": test_result["local_global_branch_weights"],
            })
            row = {
                "run": run_idx + 1,
                "seed": run_seed,
                "best_eval_oa": train_result["best_oa"],
                "test_oa": m["oa"],
                "test_aa": m["aa"],
                "test_kappa": m["kappa"],
                "test_macro_f1": m["macro_f1"],
                "output_dir": train_result["output_dir"],
            }

        elif args.mode == "test":
            test_result = run_test(args, run_seed=run_seed)
            m = test_result["metrics"]
            all_local_global_branch_weights.append({
                "run": run_idx + 1,
                "seed": run_seed,
                "result": test_result["local_global_branch_weights"],
            })
            row = {
                "run": run_idx + 1,
                "seed": run_seed,
                "test_oa": m["oa"],
                "test_aa": m["aa"],
                "test_kappa": m["kappa"],
                "test_macro_f1": m["macro_f1"],
                "output_dir": test_result["output_dir"],
            }

        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        all_rows.append(row)

    # Save run summary. This is saved even when NUM_RUNS = 1; std will be 0.
    if args.output_dir:
        summary_dir = Path(args.output_dir)
    elif args.num_runs > 1:
        summary_dir = PROJECT_ROOT / "outputs" / f"{args.dataset}_sadrfnet_lg_multirun"
    else:
        summary_dir = Path(all_rows[0]["output_dir"])

    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = summary_dir / "run_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    numeric_keys = [k for k in all_rows[0] if k not in {"run", "seed", "output_dir"}]
    summary = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in all_rows if key in row]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = 0.0 if len(values) < 2 else float(np.std(values, ddof=1))

    save_json(summary_dir / "run_summary_mean_std.json", summary)

    print("=" * 80)
    print("[Run Summary]")
    print(f"Saved run summary: {summary_csv}")
    print(f"Saved mean/std: {summary_dir / 'run_summary_mean_std.json'}")

    # Print the most important paper-style metrics.
    if "test_oa_mean" in summary:
        print(
            "Test metrics: "
            f"OA={summary['test_oa_mean']:.6f} ± {summary['test_oa_std']:.6f}, "
            f"AA={summary['test_aa_mean']:.6f} ± {summary['test_aa_std']:.6f}, "
            f"Kappa={summary['test_kappa_mean']:.6f} ± {summary['test_kappa_std']:.6f}"
        )
    elif "best_oa_mean" in summary:
        print(
            "Best validation/test-selection metric: "
            f"OA={summary['best_oa_mean']:.6f} ± {summary['best_oa_std']:.6f}"
        )
    print("=" * 80)


    # ---------------------------------------------------------------------
    # Print model.get_local_global_branch_weights() at the very end.
    # No additional output file is created.
    # ---------------------------------------------------------------------
    if all_local_global_branch_weights:
        print()
        print("=" * 80)
        print("[Local-Global Branch Weights]")
        print(
            "Source: model.get_local_global_branch_weights() "
            "(latest forward batch of each test run)"
        )

        for item in all_local_global_branch_weights:
            print(f"[Run {item['run']}, seed={item['seed']}]")
            result = item["result"]

            if not bool(result.get("available", False)):
                print(f"  {result.get('message', 'Weights unavailable.')}")
                continue

            weights = result.get("weights", {})
            if isinstance(weights, dict):
                for stage_name, stage_weights in weights.items():
                    if isinstance(stage_weights, dict):
                        local_value = stage_weights.get("local")
                        global_value = stage_weights.get("global")
                        if local_value is not None or global_value is not None:
                            print(
                                f"  {stage_name}: "
                                f"Local={local_value}, Global={global_value}"
                            )
                        else:
                            print(f"  {stage_name}: {stage_weights}")
                    else:
                        print(f"  {stage_name}: {stage_weights}")
            else:
                print(f"  {weights}")

        print("=" * 80)



# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_args_from_user_config()
    print_user_config(args)

    if args.deterministic and args.num_workers > 0:
        print("[Reproducibility Warning] DETERMINISTIC=True but NUM_WORKERS > 0.")
        print("[Reproducibility Warning] For the strongest repeatability, set NUM_WORKERS = 0.")

    if args.mode == "test" and not args.checkpoint:
        print("[Info] CHECKPOINT_PATH is empty. main.py will try to locate default outputs/.../best.pth.")

    run_multiple(args)


if __name__ == "__main__":
    main()
