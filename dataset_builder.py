"""
dataset_builder.py

Place this file directly under SADRFNet_project/.

Functions:
- scan class-folder datasets, such as datasets/AID, datasets/NWPU-RESISC45,
  datasets/UCMerced_LandUse
- create stratified train/val/test split files
- build PyTorch Dataset and DataLoader

Common train ratios:
- AID: 0.20 or 0.50, default 0.20
- NWPU-RESISC45: 0.10 or 0.20, default 0.20
- UCMerced_LandUse: 0.50 or 0.80, default 0.80
"""

from __future__ import annotations

import argparse
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass(frozen=True)
class DatasetPreset:
    folder_name: str
    common_train_ratios: Tuple[float, ...]
    default_train_ratio: float
    image_size: int = 224


DATASET_PRESETS: Dict[str, DatasetPreset] = {
    "AID": DatasetPreset("AID", (0.20, 0.50), 0.20),
    "NWPU-RESISC45": DatasetPreset("NWPU-RESISC45", (0.10, 0.20), 0.20),
    "UCMerced_LandUse": DatasetPreset("UCMerced_LandUse", (0.50, 0.80), 0.80),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def get_dataset_root(dataset_name: str, project_root: Optional[Path] = None) -> Path:
    if dataset_name not in DATASET_PRESETS:
        raise ValueError(f"Unsupported dataset_name={dataset_name}. Supported: {list(DATASET_PRESETS)}")
    project_root = get_project_root() if project_root is None else Path(project_root)
    return project_root / "datasets" / DATASET_PRESETS[dataset_name].folder_name


def get_split_dir(dataset_name: str, train_ratio: float, seed: int,
                  project_root: Optional[Path] = None) -> Path:
    project_root = get_project_root() if project_root is None else Path(project_root)
    tag = f"train{int(round(train_ratio * 100)):02d}_seed{seed}"
    return project_root / "splits" / dataset_name / tag


def scan_class_folders(dataset_root: Path) -> Tuple[List[str], Dict[str, int]]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    class_names = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class folders found under: {dataset_root}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_idx


def collect_samples(dataset_root: Path,
                    class_names: Sequence[str],
                    class_to_idx: Dict[str, int]) -> Dict[str, List[Tuple[str, int]]]:
    samples_by_class: Dict[str, List[Tuple[str, int]]] = {}
    for class_name in class_names:
        class_dir = Path(dataset_root) / class_name
        label = class_to_idx[class_name]
        samples = []
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rel_path = img_path.relative_to(dataset_root).as_posix()
                samples.append((rel_path, label))
        if not samples:
            raise RuntimeError(f"No images found in class folder: {class_dir}")
        samples_by_class[class_name] = samples
    return samples_by_class


def write_split(path: Path, samples: Sequence[Tuple[str, int]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for rel_path, label in samples:
            f.write(f"{rel_path} {label}\n")


def write_classes(path: Path, class_names: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx} {name}\n")


def read_classes(path: Path) -> List[str]:
    names = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                _, name = line.split(maxsplit=1)
                names.append(name)
    return names


def create_stratified_split(
    dataset_name: str,
    train_ratio: Optional[float] = None,
    val_ratio_within_train: float = 0.0,
    seed: int = 42,
    overwrite: bool = False,
    project_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Create per-class stratified train/val/test split.

    train_ratio:
        Benchmark training ratio. The remaining samples are used as test.
    val_ratio_within_train:
        Validation ratio split from the training pool. Use 0.0 for strict
        train/test benchmark settings.
    """
    if dataset_name not in DATASET_PRESETS:
        raise ValueError(f"Unsupported dataset_name={dataset_name}. Supported: {list(DATASET_PRESETS)}")
    if not 0 < (train_ratio or DATASET_PRESETS[dataset_name].default_train_ratio) < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 <= val_ratio_within_train < 1:
        raise ValueError("val_ratio_within_train must be in [0, 1).")

    preset = DATASET_PRESETS[dataset_name]
    train_ratio = preset.default_train_ratio if train_ratio is None else train_ratio
    project_root = get_project_root() if project_root is None else Path(project_root)
    dataset_root = get_dataset_root(dataset_name, project_root)
    split_dir = get_split_dir(dataset_name, train_ratio, seed, project_root)

    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    test_file = split_dir / "test.txt"
    classes_file = split_dir / "classes.txt"

    if train_file.exists() and test_file.exists() and classes_file.exists() and not overwrite:
        print(f"[Split] Existing split found: {split_dir}")
        return {"split_dir": split_dir, "train": train_file, "val": val_file,
                "test": test_file, "classes": classes_file}

    class_names, class_to_idx = scan_class_folders(dataset_root)
    samples_by_class = collect_samples(dataset_root, class_names, class_to_idx)
    rng = random.Random(seed)

    train_samples, val_samples, test_samples = [], [], []
    for class_name in class_names:
        samples = samples_by_class[class_name][:]
        rng.shuffle(samples)
        n_total = len(samples)
        n_train_pool = int(round(n_total * train_ratio))
        n_train_pool = max(1, min(n_train_pool, n_total - 1))

        train_pool = samples[:n_train_pool]
        test_part = samples[n_train_pool:]

        if val_ratio_within_train > 0:
            n_val = int(round(len(train_pool) * val_ratio_within_train))
            n_val = max(1, min(n_val, len(train_pool) - 1))
            val_part = train_pool[:n_val]
            train_part = train_pool[n_val:]
        else:
            val_part = []
            train_part = train_pool

        train_samples.extend(train_part)
        val_samples.extend(val_part)
        test_samples.extend(test_part)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    write_split(train_file, train_samples)
    write_split(val_file, val_samples)
    write_split(test_file, test_samples)
    write_classes(classes_file, class_names)

    print(f"[Split] Dataset: {dataset_name}")
    print(f"[Split] Dataset root: {dataset_root}")
    print(f"[Split] Output dir: {split_dir}")
    print(f"[Split] Classes: {len(class_names)}")
    print(f"[Split] Train ratio: {train_ratio}")
    print(f"[Split] Val ratio within train: {val_ratio_within_train}")
    print(f"[Split] Train/Val/Test: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")

    return {"split_dir": split_dir, "train": train_file, "val": val_file,
            "test": test_file, "classes": classes_file}


class RemoteSensingSceneDataset(Dataset):
    def __init__(self, dataset_root: Path, split_file: Path, transform=None, return_path: bool = False):
        self.dataset_root = Path(dataset_root)
        self.split_file = Path(split_file)
        self.transform = transform
        self.return_path = return_path

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {self.split_file}")

        self.samples = []
        with self.split_file.open("r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid line {line_id} in {self.split_file}: {line}")
                rel_path, label = parts
                self.samples.append((rel_path, int(label)))

        if not self.samples:
            raise RuntimeError(f"No samples found in split file: {self.split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        img_path = self.dataset_root / rel_path
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, label, rel_path
        return image, label


def build_transforms(image_size: int = 224, is_train: bool = True):
    """
    Build transforms for remote-sensing scene classification.

    The previous RandomResizedCrop(scale=(0.7, 1.0)) was too aggressive for
    small datasets such as UC Merced: it can remove discriminative scene cues
    near image borders. This version preserves more of the scene while keeping
    standard flip and mild color perturbation augmentation.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.88, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.005)
            ], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Use the full image during evaluation. For UC Merced (256x256), this avoids
    # losing border information through center cropping.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_scene_dataloaders(
    dataset_name: str = "NWPU-RESISC45",
    train_ratio: Optional[float] = None,
    val_ratio_within_train: float = 0.0,
    seed: int = 42,
    image_size: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    overwrite_split: bool = False,
    project_root: Optional[Path] = None,
) -> Dict[str, object]:
    preset = DATASET_PRESETS[dataset_name]
    train_ratio = preset.default_train_ratio if train_ratio is None else train_ratio
    image_size = preset.image_size if image_size is None else image_size
    project_root = get_project_root() if project_root is None else Path(project_root)

    split_paths = create_stratified_split(
        dataset_name=dataset_name,
        train_ratio=train_ratio,
        val_ratio_within_train=val_ratio_within_train,
        seed=seed,
        overwrite=overwrite_split,
        project_root=project_root,
    )

    dataset_root = get_dataset_root(dataset_name, project_root)
    class_names = read_classes(split_paths["classes"])

    train_dataset = RemoteSensingSceneDataset(
        dataset_root, split_paths["train"], transform=build_transforms(image_size, True)
    )
    test_dataset = RemoteSensingSceneDataset(
        dataset_root, split_paths["test"], transform=build_transforms(image_size, False)
    )

    val_dataset = None
    if split_paths["val"].exists() and split_paths["val"].stat().st_size > 0:
        val_dataset = RemoteSensingSceneDataset(
            dataset_root, split_paths["val"], transform=build_transforms(image_size, False)
        )

    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
        )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    return {
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "split_dir": split_paths["split_dir"],
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


# -------------------------------------------------------------------------
# Quick switch area
# -------------------------------------------------------------------------
# Method 1: change these variables, then run:
#     python dataset_builder.py
#
# Method 2: do not change this file; use command-line arguments:
#     python dataset_builder.py --dataset AID --train-ratio 0.2
#     python dataset_builder.py --dataset NWPU-RESISC45 --train-ratio 0.2
#     python dataset_builder.py --dataset UCMerced_LandUse --train-ratio 0.8

DEFAULT_DATASET_NAME = "NWPU-RESISC45"
DEFAULT_TRAIN_RATIO = None       # None means use default ratio in DATASET_PRESETS.
DEFAULT_VAL_RATIO_WITHIN_TRAIN = 0.0
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME,
                        choices=list(DATASET_PRESETS.keys()))
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio-within-train", type=float,
                        default=DEFAULT_VAL_RATIO_WITHIN_TRAIN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--overwrite-split", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = build_scene_dataloaders(
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overwrite_split=args.overwrite_split,
    )

    images, labels = next(iter(data["train_loader"]))
    print("[Check] Dataset:", data["dataset_name"])
    print("[Check] Dataset root:", data["dataset_root"])
    print("[Check] Split dir:", data["split_dir"])
    print("[Check] Num classes:", data["num_classes"])
    print("[Check] Train samples:", len(data["train_dataset"]))
    print("[Check] Val samples:", 0 if data["val_dataset"] is None else len(data["val_dataset"]))
    print("[Check] Test samples:", len(data["test_dataset"]))
    print("[Check] One batch images:", tuple(images.shape))
    print("[Check] One batch labels:", tuple(labels.shape))


if __name__ == "__main__":
    main()
