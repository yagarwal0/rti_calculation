"""
dataloader.py — datasets for the complete pruning experiment.

Provides train/test DataLoaders for:
    * CIFAR-10        (10 classes,  32x32 native)
    * CIFAR-100       (100 classes, 32x32 native)
    * Tiny ImageNet   (200 classes, 64x64 native -> RESIZED TO 32x32 here)

Tiny ImageNet is downsampled to 32x32 with torchvision `transforms.Resize((32, 32))`
so every dataset feeds the network 3x32x32 inputs — i.e. the same models used
for CIFAR can be reused directly (handy for cifar->tinyimagenet transfer).

Unified entry point
-------------------
    from dataloader import get_loaders

    train_loader, test_loader, num_classes = get_loaders(
        'cifar10',                 # 'cifar10' | 'cifar100' | 'tinyimagenet'
        batch_size=128,
        data_root='./data',                       # CIFAR download/cache dir
        tinyimagenet_dir='./tiny-imagenet-200',   # pre-downloaded TinyImageNet
    )

Tiny ImageNet must be downloaded and unzipped beforehand (it is not on
torchvision):  http://cs231n.stanford.edu/tiny-imagenet-200.zip
Expected layout:
    tiny-imagenet-200/
    ├── train/<wnid>/images/*.JPEG
    ├── val/images/*.JPEG
    └── val/val_annotations.txt
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image


# ---------------------------------------------------------------------------
# Per-dataset normalization statistics
# ---------------------------------------------------------------------------
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
# Tiny ImageNet's own channel statistics (computed on the 64x64 train split).
TINYIMAGENET_MEAN, TINYIMAGENET_STD = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------
def get_cifar10_loaders(batch_size=128, num_workers=2, data_root='./data',
                        download=True, normalize=False):
    """Returns (train_loader, test_loader, num_classes) for CIFAR-10 (32x32).

    `normalize=False` (default) loads images in [0, 1] with NO Normalize step,
    matching the attack convention (FGSM/PGD/AutoAttack): per-channel
    normalization is applied *inside the model* via `attacks.NormalizedModel`.
    Set `normalize=True` to fold CIFAR-10 standardization into the transform.
    """
    train_tfms = [transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor()]
    test_tfms = [transforms.ToTensor()]
    if normalize:
        train_tfms.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
        test_tfms.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    transform_train = transforms.Compose(train_tfms)
    transform_test = transforms.Compose(test_tfms)

    trainset = CIFAR10(root=data_root, train=True, download=download, transform=transform_train)
    testset = CIFAR10(root=data_root, train=False, download=download, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print("CIFAR-10 loaded (32x32, 10 classes).")
    return train_loader, test_loader, 10


# ---------------------------------------------------------------------------
# CIFAR-100
# ---------------------------------------------------------------------------
def get_cifar100_loaders(batch_size=128, num_workers=2, data_root='./data',
                         download=True, normalize=False):
    """Returns (train_loader, test_loader, num_classes) for CIFAR-100 (32x32).

    `normalize=False` (default) loads images in [0, 1] with NO Normalize step
    (normalization is applied inside the model via `attacks.NormalizedModel`).
    Set `normalize=True` to fold CIFAR-100 standardization into the transform.
    """
    train_tfms = [transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor()]
    test_tfms = [transforms.ToTensor()]
    if normalize:
        train_tfms.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
        test_tfms.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    transform_train = transforms.Compose(train_tfms)
    transform_test = transforms.Compose(test_tfms)

    trainset = CIFAR100(root=data_root, train=True, download=download, transform=transform_train)
    testset = CIFAR100(root=data_root, train=False, download=download, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print("CIFAR-100 loaded (32x32, 100 classes).")
    return train_loader, test_loader, 100


# ---------------------------------------------------------------------------
# Tiny ImageNet validation set (flat dir + annotations file)
# ---------------------------------------------------------------------------
class TinyImageNetValDataset(Dataset):
    """Tiny ImageNet validation split: images live in a flat folder and labels
    are listed in `val_annotations.txt`. Uses the train split's class_to_idx so
    integer labels match between train and val."""

    def __init__(self, val_dir, class_to_idx, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_dir = os.path.join(val_dir, 'images')
        self.annotations_path = os.path.join(val_dir, 'val_annotations.txt')
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        with open(self.annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name, class_id = parts[0], parts[1]
                img_path = os.path.join(self.image_dir, img_name)
                label = self.class_to_idx[class_id]
                samples.append((img_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Tiny ImageNet — resized to 32x32
# ---------------------------------------------------------------------------
def get_tinyimagenet_loaders(batch_size=128, data_dir='./tiny-imagenet-200',
                             num_workers=4, norm='tinyimagenet', normalize=False):
    """Returns (train_loader, test_loader, num_classes) for Tiny ImageNet,
    with every image RESIZED TO 32x32 via torchvision transforms.

    Args:
        batch_size (int)
        data_dir (str): root of the unzipped tiny-imagenet-200 folder.
        num_workers (int)
        norm (str): which channel statistics to use *if* `normalize=True` —
            'tinyimagenet' (default) -> Tiny ImageNet's own channel stats,
            'cifar'                   -> CIFAR-10 stats.
        normalize (bool): if False (default) images are loaded in [0, 1] with NO
            Normalize step (normalization is applied inside the model via
            `attacks.NormalizedModel`); if True the chosen stats are folded into
            the transform.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Tiny ImageNet directory not found at '{data_dir}'.\n"
            "Download and unzip it from "
            "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        )

    # 64x64 -> 32x32, then standard CIFAR-style augmentation.
    train_tfms = [transforms.Resize((32, 32)),
                  transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor()]
    test_tfms = [transforms.Resize((32, 32)), transforms.ToTensor()]
    if normalize:
        mean, std = (CIFAR10_MEAN, CIFAR10_STD) if norm == 'cifar' \
            else (TINYIMAGENET_MEAN, TINYIMAGENET_STD)
        train_tfms.append(transforms.Normalize(mean, std))
        test_tfms.append(transforms.Normalize(mean, std))
    train_transform = transforms.Compose(train_tfms)
    test_transform = transforms.Compose(test_tfms)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    num_classes = len(train_dataset.classes)
    class_to_idx = train_dataset.class_to_idx

    test_dataset = TinyImageNetValDataset(val_dir, class_to_idx=class_to_idx,
                                          transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Tiny ImageNet loaded (resized to 32x32, {num_classes} classes): "
          f"{len(train_dataset)} train / {len(test_dataset)} test samples.")
    return train_loader, test_loader, num_classes


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------
def get_loaders(dataset, batch_size=128, num_workers=2, data_root='./data',
                tinyimagenet_dir='./tiny-imagenet-200', download=True,
                tinyimagenet_norm='tinyimagenet', normalize=False):
    """Return (train_loader, test_loader, num_classes) for the named dataset.

    dataset: 'cifar10' | 'cifar100' | 'tinyimagenet' (aliases: 'tiny-imagenet',
             'tiny', 'tinyimagenet200').
    normalize: if False (default) images are loaded in [0, 1] with NO Normalize
             step — matching the FGSM/PGD/AutoAttack convention, where
             normalization is applied inside the model via
             `attacks.NormalizedModel`. Set True to standardize in the transform.
    """
    name = dataset.lower().replace('_', '').replace('-', '')
    if name == 'cifar10':
        return get_cifar10_loaders(batch_size, num_workers, data_root, download,
                                   normalize=normalize)
    if name == 'cifar100':
        return get_cifar100_loaders(batch_size, num_workers, data_root, download,
                                    normalize=normalize)
    if name in ('tinyimagenet', 'tiny', 'tinyimagenet200'):
        # Respect the caller's num_workers (use 0 on Windows to avoid the slow
        # spawn-and-reimport of every DataLoader worker).
        return get_tinyimagenet_loaders(batch_size, tinyimagenet_dir,
                                        num_workers, tinyimagenet_norm,
                                        normalize=normalize)
    raise ValueError(
        f"Unknown dataset '{dataset}'. "
        "Choose from 'cifar10', 'cifar100', 'tinyimagenet'."
    )


if __name__ == '__main__':
    # Quick shape check (CIFAR auto-downloads; TinyImageNet must exist locally).
    for ds in ('cifar10', 'cifar100'):
        tr, te, nc = get_loaders(ds, batch_size=64)
        x, y = next(iter(tr))
        print(f"{ds}: batch {tuple(x.shape)}, labels {tuple(y.shape)}, classes {nc}")
