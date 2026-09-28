"""
test.py

Independent test script for SADRFNet.

Place this file directly under SADRFNet_project/.

Example:
    python test.py --dataset UCMerced_LandUse --train-ratio 0.8 --checkpoint outputs/UCMerced_LandUse_sadrfnet_train80_seed42/best.pth
    python test.py --dataset NWPU-RESISC45 --train-ratio 0.2 --checkpoint outputs/NWPU-RESISC45_sadrfnet_train20_seed42/best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


def build_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_metrics(targets: Iterable[int], preds: Iterable[int], num_classes: int) -> Dict[str, object]:
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
# Checkpoint and evaluation
# -----------------------------------------------------------------------------


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_model_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    info = {
        "epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
        "best_acc": checkpoint.get("best_acc", None) if isinstance(checkpoint, dict) else None,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return info


def get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs


@torch.no_grad()
def forward_with_tta(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits = get_logits(model(images))
    logits = logits + get_logits(model(torch.flip(images, dims=[3])))
    logits = logits + get_logits(model(torch.flip(images, dims=[2])))
    logits = logits + get_logits(model(torch.flip(images, dims=[2, 3])))
    return logits / 4.0


def progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, ncols=100)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    num_classes: int,
    tta: bool = False,
) -> Dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []

    for images, labels in progress(loader, "Test"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = forward_with_tta(model, images) if tta else get_logits(model(images))
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        conf, preds = torch.max(probs, dim=1)

        loss_meter.update(loss.item(), labels.size(0))
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.extend(conf.cpu().numpy().tolist())

    metrics = compute_metrics(all_targets, all_preds, num_classes)
    metrics["loss"] = loss_meter.avg
    metrics["mean_confidence"] = float(np.mean(all_probs)) if all_probs else 0.0
    return metrics


def save_test_outputs(output_dir: Path, metrics: Dict[str, object], class_names: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        k: v for k, v in metrics.items()
        if k not in {"confusion_matrix", "per_class_acc"}
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently test SADRFNet.")

    # Dataset.
    parser.add_argument("--dataset", type=str, default="NWPU-RESISC45", choices=list(DATASET_PRESETS.keys()))
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio-within-train", type=float, default=0.0,
                        help="Use the same value as training if you created a validation split. Test set itself is unchanged.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Model.
    parser.add_argument("--fusion-channels", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--switch-to-deploy", action="store_true",
                        help="After loading the training checkpoint, fuse DRB branches for deploy-style testing.")
    parser.add_argument("--test-tta", action="store_true",
                        help="Average logits over original / horizontal / vertical / hv-flipped inputs.")

    # Runtime.
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--allow-partial-load", action="store_true",
                        help="Use strict=False when loading checkpoint. Not recommended unless model definition changed.")
    parser.add_argument("--output-dir", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[Runtime] Device: {device}")

    data = build_scene_dataloaders(
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite_split=False,
        project_root=PROJECT_ROOT,
    )
    test_loader = data["test_loader"]
    num_classes = int(data["num_classes"])
    class_names = data["class_names"]

    print(f"[Data] Dataset: {data['dataset_name']}")
    print(f"[Data] Dataset root: {data['dataset_root']}")
    print(f"[Data] Split dir: {data['split_dir']}")
    print(f"[Data] Classes: {num_classes}")
    print(f"[Data] Test samples: {len(data['test_dataset'])}")

    model = sadrfnet50(
        num_classes=num_classes,
        in_chans=3,
        fusion_channels=args.fusion_channels,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        return_features=False,
        deploy=False,
    ).to(device)

    load_info = load_model_checkpoint(
        Path(args.checkpoint),
        model,
        device=device,
        strict=not args.allow_partial_load,
    )
    print(f"[Checkpoint] Loaded: {args.checkpoint}")
    print(f"[Checkpoint] Epoch: {load_info['epoch']}, best_acc: {load_info['best_acc']}")
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        print(f"[Checkpoint] Missing keys: {load_info['missing_keys']}")
        print(f"[Checkpoint] Unexpected keys: {load_info['unexpected_keys']}")

    if args.switch_to_deploy:
        model.eval()
        model.switch_to_deploy()
        print("[Deploy] DRB branches have been fused by model.switch_to_deploy().")

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device, num_classes, tta=bool(args.test_tta))

    if not args.output_dir:
        checkpoint_path = Path(args.checkpoint)
        args.output_dir = str(checkpoint_path.parent / "test_results")
    output_dir = Path(args.output_dir)
    save_test_outputs(output_dir, metrics, class_names)

    print("[Test Results]")
    print(f"  Loss            : {metrics['loss']:.6f}")
    print(f"  OA / Top-1 Acc  : {metrics['oa']:.6f}")
    print(f"  AA              : {metrics['aa']:.6f}")
    print(f"  Macro Precision : {metrics['macro_precision']:.6f}")
    print(f"  Macro Recall    : {metrics['macro_recall']:.6f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.6f}")
    print(f"  Kappa           : {metrics['kappa']:.6f}")
    print(f"  Mean Confidence : {metrics['mean_confidence']:.6f}")
    print(f"[Saved] {output_dir}")


if __name__ == "__main__":
    main()
