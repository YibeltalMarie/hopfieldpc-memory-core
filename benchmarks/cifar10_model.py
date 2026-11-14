"""CIFAR-10 backbone: encoder (L1-L7 analogue), shared head and training augmentation.

The network reuses ``mnist_baseline.MnistNet`` (encoder -> bridge -> head) and
``ClassifierHead``, so every memory benchmark treats CIFAR-10 runs exactly
like MNIST runs. Encoders take raw images in [0, 1] and normalize inside the
forward pass with per-channel statistics stored as buffers; the statistics
come from the training split only and are saved with the checkpoint.

Architectures (``model.name``):

    cnn       three conv blocks [conv-BN-ReLU x 2, max-pool] with widths
              ``model.widths``, global average pooling, linear to ``embed_dim``
    resnet18  CIFAR ResNet-18 (3x3 stem, no stem pooling; He et al. 2016),
              global average pooling, linear to ``embed_dim``

The embedding layer has no activation (signed z), as in the MNIST encoder.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from benchmarks.mnist_baseline import ClassifierHead, MnistNet

ARCHITECTURES = ("cnn", "resnet18")
DEFAULT_MODEL = {"name": "cnn", "widths": [32, 64, 128], "embed_dim": 64}
CROP_PADDING = 4


class _Normalized(nn.Module):
    """Holds per-channel mean/std buffers [1, C, 1, 1] and normalizes inputs."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.shape != std.shape or mean.ndim != 1 or (std <= 0).any():
            raise ValueError("mean and std must be positive-std vectors [C] of equal shape.")
        self.register_buffer("mean", mean.float().view(1, -1, 1, 1).clone())
        self.register_buffer("std", std.float().view(1, -1, 1, 1).clone())

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


class CifarCNN(_Normalized):
    """images [B, 3, 32, 32] in [0, 1] -> z [B, embed_dim]."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, *, widths: list[int], embed_dim: int) -> None:
        super().__init__(mean, std)
        layers: list[nn.Module] = []
        c_in = mean.numel()
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.embed = nn.Linear(c_in, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(self.normalize(images))
        return self.embed(x.mean(dim=(2, 3)))


class _BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class CifarResNet18(_Normalized):
    """CIFAR ResNet-18: images [B, 3, 32, 32] in [0, 1] -> z [B, embed_dim]."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, *, embed_dim: int) -> None:
        super().__init__(mean, std)
        self.stem = nn.Sequential(nn.Conv2d(mean.numel(), 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True))
        blocks, c_in = [], 64
        for c_out, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            blocks += [_BasicBlock(c_in, c_out, stride), _BasicBlock(c_out, c_out, 1)]
            c_in = c_out
        self.layers = nn.Sequential(*blocks)
        self.embed = nn.Linear(512, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layers(self.stem(self.normalize(images)))
        return self.embed(x.mean(dim=(2, 3)))


def validate_model_config(model: Any) -> dict[str, Any]:
    """Normalized model config: {'name', 'embed_dim'} (+ 'widths' for cnn)."""
    if model is None:
        model = DEFAULT_MODEL
    if not isinstance(model, dict) or model.get("name") not in ARCHITECTURES:
        raise ValueError(f"model.name must be one of {ARCHITECTURES}, got {model!r}.")
    allowed = {"name", "embed_dim"} | ({"widths"} if model["name"] == "cnn" else set())
    unknown = sorted(set(model) - allowed)
    if unknown:
        raise ValueError(f"Unknown model keys for {model['name']}: {unknown}.")
    embed_dim = model.get("embed_dim", 64)
    if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim < 1:
        raise ValueError(f"model.embed_dim must be a positive integer, got {embed_dim!r}.")
    out: dict[str, Any] = {"name": model["name"], "embed_dim": embed_dim}
    if model["name"] == "cnn":
        widths = model.get("widths", DEFAULT_MODEL["widths"])
        if (not isinstance(widths, list) or not widths
                or any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in widths)):
            raise ValueError(f"model.widths must be a non-empty list of positive integers, got {widths!r}.")
        if 32 % (2 ** len(widths)) != 0:
            raise ValueError("model.widths has too many blocks for 32 x 32 images.")
        out["widths"] = list(widths)
    return out


def build_cifar10_net(seed: int, model: dict[str, Any], mean: torch.Tensor, std: torch.Tensor,
                      num_classes: int = 10) -> MnistNet:
    """Baseline network (bridge = Identity) with seeded initialization."""
    model = validate_model_config(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if model["name"] == "cnn":
            encoder: nn.Module = CifarCNN(mean, std, widths=model["widths"], embed_dim=model["embed_dim"])
        else:
            encoder = CifarResNet18(mean, std, embed_dim=model["embed_dim"])
        head = ClassifierHead(embed_dim=model["embed_dim"], num_classes=num_classes)
    return MnistNet(encoder, nn.Identity(), head)


def augment_batch(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random crop (zero padding 4) and horizontal flip of a training batch [B, C, H, W].

    Used only while training the backbone; embeddings for memory construction
    and evaluation are always computed from unaugmented images.
    """
    b, _, h, w = images.shape
    padded = F.pad(images, (CROP_PADDING,) * 4)
    tops = torch.randint(0, 2 * CROP_PADDING + 1, (b,), generator=generator).tolist()
    lefts = torch.randint(0, 2 * CROP_PADDING + 1, (b,), generator=generator).tolist()
    flips = (torch.rand(b, generator=generator) < 0.5).tolist()
    out = torch.empty_like(images)
    for i in range(b):
        crop = padded[i, :, tops[i]:tops[i] + h, lefts[i]:lefts[i] + w]
        out[i] = crop.flip(-1) if flips[i] else crop
    return out
