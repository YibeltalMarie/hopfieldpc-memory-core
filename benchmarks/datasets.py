"""Dataset registry shared by every benchmark.

Each entry describes one image-classification dataset and how to load it as
``(train_images, train_labels, test_images, test_labels)``: images are float
tensors [N, C, H, W] in [0, 1] (no mean/std normalization; encoders normalize
inside their forward pass), labels are long tensors [N] in [0, num_classes).

    mnist    [N, 1, 28, 28], delegates to benchmarks.mnist_baseline.load_mnist
    cifar10  [N, 3, 32, 32], torchvision CIFAR-10 (50 000 train / 10 000 test)

Normalization statistics, when an encoder needs them, come from
``channel_stats`` applied to the training split only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.mnist_baseline import load_mnist

Loader = Callable[..., "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]"]


def load_cifar10(data_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 as (train_images, train_labels, test_images, test_labels).

    torchvision stores images as uint8 [N, 32, 32, 3]; they are returned as
    float [N, 3, 32, 32] in [0, 1].
    """
    import torchvision

    out = []
    for train in (True, False):
        ds = torchvision.datasets.CIFAR10(root=str(data_dir), train=train, download=True)
        images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255.0).contiguous()
        out += [images, torch.tensor(ds.targets, dtype=torch.long)]
    return out[0], out[1], out[2], out[3]


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one dataset."""

    name: str
    num_classes: int
    channels: int
    image_size: int
    loader: Loader


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", num_classes=10, channels=1, image_size=28, loader=load_mnist),
    "cifar10": DatasetSpec("cifar10", num_classes=10, channels=3, image_size=32, loader=load_cifar10),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}.")
    return DATASETS[name]


def check_tensors(spec: DatasetSpec, images: torch.Tensor, labels: torch.Tensor) -> None:
    """Raise if a split does not match the dataset's shape, range and label contract."""
    expected = (spec.channels, spec.image_size, spec.image_size)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected:
        raise ValueError(f"{spec.name}: images must be [N, {', '.join(map(str, expected))}], "
                         f"got {tuple(images.shape)}.")
    if not images.is_floating_point() or images.min() < 0 or images.max() > 1:
        raise ValueError(f"{spec.name}: images must be floats in [0, 1].")
    if labels.dtype != torch.long or labels.shape != images.shape[:1]:
        raise ValueError(f"{spec.name}: labels must be a long tensor [N] matching the images.")
    if labels.numel() and (labels.min() < 0 or labels.max() >= spec.num_classes):
        raise ValueError(f"{spec.name}: labels must lie in [0, {spec.num_classes}).")


def load_dataset(name: str, data_dir: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and check (train_images, train_labels, test_images, test_labels)."""
    spec = get_dataset(name)
    data = spec.loader(data_dir)
    check_tensors(spec, data[0], data[1])
    check_tensors(spec, data[2], data[3])
    return data


def channel_stats(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std [C] of images [N, C, H, W], accumulated in float64.

    Call it on the training split only, so validation and test data never
    influence normalization.
    """
    x = images.double().transpose(0, 1).reshape(images.shape[1], -1)
    return x.mean(dim=1).float(), x.std(dim=1).float()
