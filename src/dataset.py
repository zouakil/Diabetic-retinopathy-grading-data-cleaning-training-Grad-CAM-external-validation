"""
dataset.py - PyTorch data pipeline: loading, augmentation and class balancing.

PURPOSE
    Turn the preprocessed images (data/processed_512) and the split lists
    (data/splits/train.csv and val.csv) into batches of tensors for the model.

CONTENTS
    get_transforms         training: random shift/zoom, flips, rotation, color jitter,
                           then normalization; validation: normalization only
    RetinaDataset          returns (image tensor [3, 512, 512], grade) for one row
    make_weighted_sampler  draws rare grades more often; weight = 1 / count**power
                           (power 0.5 = softer than a fully balanced epoch)
    get_dataloaders        builds the training and validation loaders

NOTES
    - The sampler changes WHICH images are picked, not the images themselves.
    - The validation loader is never augmented or resampled, so metrics reflect
      the real class distribution.
    - Paths are hard-coded to the APTOS layout (id_code, diagnosis, .png).
      External datasets need different column and extension settings.
    - Sampling and augmentation are random, so runs vary slightly even with
      fixed seeds.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2

# Paths are computed from this file's location, so the code works
# no matter where you run it from (project root or notebooks/).
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Values the pretrained EfficientNet expects (ImageNet statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool, size: int = 512):
    """Training gets random augmentation, validation gets none."""
    if train:
        return v2.Compose([
            v2.ToImage(),                                # numpy (H,W,3) -> tensor (3,H,W)
            v2.ToDtype(torch.float32, scale=True),       # 0..255 -> 0..1
            v2.Resize((size, size), antialias=True),
            v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=20),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((size, size), antialias=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RetinaDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.img_dir / f"{row['id_code']}.png"

        img = cv2.imread(str(path))
        assert img is not None, f"Could not read image: {path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = int(row["diagnosis"])
        return img, label

    @property
    def labels(self):
        return self.df["diagnosis"].values


def make_weighted_sampler(labels, power=0.5):
    """power=1.0 -> fully balanced, power=0.5 -> softer, power=0 -> no balancing."""
    labels = np.asarray(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts ** power
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def get_dataloaders(batch_size=16, size=512, num_workers=4, use_sampler=True, sampler_power=0.5):
    img_dir = DATA / "processed_512"
    train_ds = RetinaDataset(DATA / "splits" / "train.csv", img_dir, get_transforms(True, size))
    val_ds = RetinaDataset(DATA / "splits" / "val.csv", img_dir, get_transforms(False, size))

    sampler = make_weighted_sampler(train_ds.labels, power=sampler_power) if use_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),     # sampler and shuffle can't be used together
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_ds, val_ds