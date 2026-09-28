"""
train.py

Place this file directly under SADRFNet_project/.

Expected project structure:
    SADRFNet_project/
    ├── train.py
    ├── test.py
    ├── dataset_builder.py
    ├── datasets/
    │   ├── AID/
    │   ├── NWPU-RESISC45/
    │   └── UCMerced_LandUse/
    └── models/
        └── sadrfnet.py

Example:
    python train.py --dataset UCMerced_LandUse --train-ratio 0.8 --epochs 100 --batch-size 16
    python train.py --dataset NWPU-RESISC45 --train-ratio 0.2 --epochs 100 --batch-size 32
    python train.py --dataset AID --train-ratio 0.2 --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_builder import DATASET_PRESETS, build_scene_dataloaders  # noqa: E402
from models.sadrfnet import sadrfnet50  # noqa: E402


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
        self.sum += float(value) * n
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def build_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_metrics(
    targets: Iterable[int],
    preds: Iterable[int],
    num_classes: int,
) -> Dict[str, object]:
    targets = np.asarray(list(targets), dtype=np.int64)
    preds = np.asarray(list(preds), dtype=np.int64)
    cm = build_confusion_matrix(targets, preds, num_classes)

    total = cm.sum()
    correct = np.trace(cm)
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
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "kappa": kappa,
        "per_class_acc": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
    }


# -----------------------------------------------------------------------------
# Model / optimizer / checkpoint
# -----------------------------------------------------------------------------


def build_model(num_classes: int, args: argparse.Namespace) -> nn.Module:
    model = sadrfnet50(
        num_classes=num_classes,
        in_chans=3,
        fusion_channels=args.fusion_channels,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        return_features=False,
        deploy=False,
        pretrained=bool(args.use_imagenet_pretrained),
        pretrained_path=args.imagenet_pretrained_path or None,
        pretrained_verbose=bool(args.imagenet_pretrained_verbose),
    )
    return model


def _is_new_or_random_module(name: str) -> bool:
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
    param_groups = {}
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new = _is_new_or_random_module(param_name)
        lr = args.lr * (args.new_module_lr_mult if is_new else args.backbone_lr_mult)
        if args.no_weight_decay_on_norm_bias and _is_norm_or_bias(param_name, param):
            wd = 0.0
        else:
            wd = args.weight_decay
        key = (float(lr), float(wd))
        if key not in param_groups:
            param_groups[key] = {"params": [], "lr": float(lr), "weight_decay": float(wd)}
        param_groups[key]["params"].append(param)
    groups = list(param_groups.values())

    if args.optimizer.lower() == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if args.optimizer.lower() == "sgd":
        return torch.optim.SGD(groups, momentum=args.momentum, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(optimizer, args: argparse.Namespace):
    scheduler_name = args.scheduler.lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    metrics: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
) -> Tuple[int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1
    best_acc = float(checkpoint.get("best_acc", 0.0)) if isinstance(checkpoint, dict) else 0.0

    if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and isinstance(checkpoint, dict) and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return start_epoch, best_acc


# -----------------------------------------------------------------------------
# Train / evaluate
# -----------------------------------------------------------------------------


def get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs


def progress(iterable, desc: str):
    if tqdm is None:
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
        loss_meter.update(loss.item(), batch_size)
        preds = torch.argmax(logits, dim=1)
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
) -> Dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, labels in progress(loader, desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = get_logits(model(images))
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), labels.size(0))
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(all_targets, all_preds, num_classes)
    metrics["loss"] = loss_meter.avg
    return metrics


def append_csv_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SADRFNet for remote-sensing scene classification.")

    # Dataset.
    parser.add_argument("--dataset", type=str, default="NWPU-RESISC45", choices=list(DATASET_PRESETS.keys()))
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Common split ratio. None means using the preset default ratio.")
    parser.add_argument("--val-ratio-within-train", type=float, default=0.0,
                        help="Set 0.1 to create a validation set from the training pool. Use 0.0 for strict train/test split.")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model.
    parser.add_argument("--fusion-channels", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-imagenet-pretrained", dest="use_imagenet_pretrained", action="store_false",
                        help="Disable partial ImageNet ResNet-50 initialization.")
    parser.set_defaults(use_imagenet_pretrained=True)
    parser.add_argument("--imagenet-pretrained-path", type=str, default="")
    parser.add_argument("--imagenet-pretrained-verbose", action="store_true")

    # Optimization.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--new-module-lr-mult", type=float, default=1.0)
    parser.add_argument("--no-weight-decay-on-norm-bias", action="store_true", default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")

    # Runtime.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="", help="Default: outputs/<dataset>_sadrfnet_trainXX_seedXX")
    parser.add_argument("--save-every", type=int, default=0, help="Save epoch checkpoint every N epochs. 0 means disabled.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    effective_train_ratio = args.train_ratio
    if effective_train_ratio is None:
        effective_train_ratio = DATASET_PRESETS[args.dataset].default_train_ratio

    if not args.output_dir:
        args.output_dir = str(
            PROJECT_ROOT / "outputs" / f"{args.dataset}_adapter_sadrfnet_train{int(effective_train_ratio * 100):02d}_seed{args.seed}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[Runtime] Device: {device}")
    print(f"[Runtime] Output dir: {output_dir}")

    data = build_scene_dataloaders(
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite_split=args.overwrite_split,
        project_root=PROJECT_ROOT,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    num_classes = int(data["num_classes"])

    if val_loader is None:
        eval_loader = test_loader
        eval_name = "test"
        print("[Warning] No validation split found. Best checkpoint will be selected on the test set.")
        print("[Warning] For paper-level strict evaluation, consider using --val-ratio-within-train 0.1 or save the last checkpoint and test once.")
    else:
        eval_loader = val_loader
        eval_name = "val"

    print(f"[Data] Dataset: {data['dataset_name']}")
    print(f"[Data] Dataset root: {data['dataset_root']}")
    print(f"[Data] Split dir: {data['split_dir']}")
    print(f"[Data] Classes: {num_classes}")
    print(f"[Data] Train/Val/Test samples: {len(data['train_dataset'])}/"
          f"{0 if data['val_dataset'] is None else len(data['val_dataset'])}/"
          f"{len(data['test_dataset'])}")

    model = build_model(num_classes, args).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(Path(args.resume), model, optimizer, scheduler, map_location=device)
        print(f"[Resume] Loaded: {args.resume}")
        print(f"[Resume] Start epoch: {start_epoch}, best acc: {best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, args)
        eval_stats = evaluate(model, eval_loader, criterion, device, num_classes, desc=f"{eval_name.capitalize()} Epoch {epoch}")

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        eval_acc = float(eval_stats["oa"])
        is_best = eval_acc > best_acc
        if is_best:
            best_acc = eval_acc

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            f"{eval_name}_loss": eval_stats["loss"],
            f"{eval_name}_oa": eval_stats["oa"],
            f"{eval_name}_aa": eval_stats["aa"],
            f"{eval_name}_macro_f1": eval_stats["macro_f1"],
            f"{eval_name}_kappa": eval_stats["kappa"],
            "best_acc": best_acc,
        }
        append_csv_row(output_dir / "train_log.csv", row)

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"lr={lr:.6g} "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
            f"{eval_name}_loss={eval_stats['loss']:.4f} {eval_name}_oa={eval_stats['oa']:.4f} "
            f"{eval_name}_aa={eval_stats['aa']:.4f} {eval_name}_f1={eval_stats['macro_f1']:.4f} "
            f"best={best_acc:.4f}"
        )

        save_checkpoint(
            output_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_acc,
            args,
            eval_stats,
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                args,
                eval_stats,
            )
            print(f"[Checkpoint] Saved best checkpoint: {output_dir / 'best.pth'}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                args,
                eval_stats,
            )

    print(f"[Done] Best {eval_name} OA: {best_acc:.4f}")
    print(f"[Done] Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
