from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"


@dataclass(frozen=True)
class Split:
    seed: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    print(f"Downloading: {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def ensure_oxford102(root: Path) -> tuple[list[Path], np.ndarray]:
    """
    Ensures Oxford102 exists under:
      root/data/oxford102/jpg
      root/data/oxford102/imagelabels.mat
    Returns:
      image_paths: sorted list of jpg paths (image_00001.jpg ...)
      labels: np.ndarray shape (N,) labels in 1..102
    """
    data_dir = root / "data" / "oxford102"
    images_dir = data_dir / "jpg"
    tgz_path = data_dir / "102flowers.tgz"
    labels_mat = data_dir / "imagelabels.mat"

    download_file(FLOWERS_URL, tgz_path)
    download_file(LABELS_URL, labels_mat)

    # Extract if needed
    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) == 0:
        print(f"Extracting {tgz_path} -> {data_dir}")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    labels = loadmat(labels_mat)["labels"].squeeze().astype(int)
    image_paths = sorted(images_dir.glob("image_*.jpg"))

    if len(image_paths) != len(labels):
        raise RuntimeError(f"Mismatch images={len(image_paths)} labels={len(labels)}")

    return image_paths, labels


def make_random_split(labels_array: np.ndarray, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    n = len(labels_array)

    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    n_train = n // 2
    n_val = n // 4
    n_test = n - n_train - n_val

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    # sanity checks
    assert len(train_idx) == n_train
    assert len(val_idx) == n_val
    assert len(test_idx) == n_test
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    return Split(seed=seed, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def split_summary(name: str, split: Split, labels_array: np.ndarray) -> None:
    def counts(idxs: np.ndarray) -> tuple[int, int]:
        y = labels_array[idxs]
        return len(idxs), int(len(np.unique(y)))

    n_tr, c_tr = counts(split.train_idx)
    n_va, c_va = counts(split.val_idx)
    n_te, c_te = counts(split.test_idx)

    print(f"{name} (seed={split.seed}):")
    print(f"  Train: {n_tr} samples | {c_tr} unique classes")
    print(f"  Val:   {n_va} samples | {c_va} unique classes")
    print(f"  Test:  {n_te} samples | {c_te} unique classes")


# ImageNet stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class Oxford102Dataset(Dataset):
    """
    Returns (image, label) where label is mapped:
      1..102  ->  0..101
    """
    def __init__(
        self,
        image_paths: list[Path],
        labels: np.ndarray,
        indices: np.ndarray,
        transform=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.indices = np.array(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = Image.open(self.image_paths[idx]).convert("RGB")
        y = int(self.labels[idx]) - 1  # 1..102 -> 0..101

        if self.transform is not None:
            img = self.transform(img)

        return img, y


def build_datasets_for_split(
    image_paths: list[Path],
    labels: np.ndarray,
    split: Split,
    transform_train,
    transform_eval
):
    ds_train = Oxford102Dataset(image_paths, labels, split.train_idx, transform=transform_train)
    ds_val = Oxford102Dataset(image_paths, labels, split.val_idx, transform=transform_eval)
    ds_test = Oxford102Dataset(image_paths, labels, split.test_idx, transform=transform_eval)
    return ds_train, ds_val, ds_test


def build_dataloaders(
    ds_train,
    ds_val,
    ds_test,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = False
):
    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return dl_train, dl_val, dl_test
