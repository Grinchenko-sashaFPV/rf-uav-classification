"""
PyTorch Dataset для RFUAV-спектрограм.
Об'єднує train/ і valid/ з RFUAV, робить власний 80/20 stratified split.
"""
from pathlib import Path
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TRAIN_DIR, VALID_DIR, CLASSES, LABEL_MAP, IMG_SIZE, VAL_SPLIT, SEED
)


def gather_all_images() -> list[tuple[Path, int]]:
    """
    Збирає всі .jpg з train/ і valid/ для обраних класів.
    Повертає список (path, label).
    """
    items: list[tuple[Path, int]] = []
    for class_name in CLASSES:
        label = LABEL_MAP[class_name]
        for split_dir in (TRAIN_DIR, VALID_DIR):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  WARN: {class_dir} не існує")
                continue
            for img_path in class_dir.glob("*.jpg"):
                items.append((img_path, label))
    return items


def stratified_split(
    items: list[tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list, list]:
    """
    Розбиття на train і val.
    Зберігає однакову частку кожного класу в обох частинах.
    """
    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for item in items:
        by_class.setdefault(item[1], []).append(item)

    train_items, val_items = [], []
    for label, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_val = int(len(group) * val_ratio)
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


# ImageNet нормалізація
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),    # спектрограма симетрична за часом — фліп ОК
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class RFUAVDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform: Optional[T.Compose] = None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_datasets() -> tuple[RFUAVDataset, RFUAVDataset]:
    """Готує train і val датасети."""
    all_items = gather_all_images()
    print(f"Знайдено {len(all_items)} зображень у {len(CLASSES)} класах")
    train_items, val_items = stratified_split(all_items, VAL_SPLIT, SEED)
    print(f"  train: {len(train_items)},  val: {len(val_items)}")
    return (
        RFUAVDataset(train_items, train_transform),
        RFUAVDataset(val_items, val_transform),
    )


if __name__ == "__main__":
    # Швидкий sanity check
    train_ds, val_ds = make_datasets()
    img, label = train_ds[0]
    print(f"\nSample: tensor shape {tuple(img.shape)}, label {label} = {CLASSES[label]}")