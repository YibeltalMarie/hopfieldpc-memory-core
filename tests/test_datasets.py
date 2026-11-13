"""Tests for the dataset registry, the dataset-aware entry points and the
dataset-agnostic hooks added to benchmarks/mnist_baseline.py.

No test downloads data: loaders are exercised with small synthetic tensors.
"""

from __future__ import annotations

import pytest
import torch

import benchmarks.mnist_baseline as mb
from benchmarks import runs
from benchmarks.datasets import DATASETS, channel_stats, check_tensors, get_dataset, load_dataset

# -- registry ----------------------------------------------------------------


def test_registry_describes_mnist_and_cifar10() -> None:
    assert (DATASETS["mnist"].channels, DATASETS["mnist"].image_size) == (1, 28)
    assert (DATASETS["cifar10"].channels, DATASETS["cifar10"].image_size) == (3, 32)
    assert all(spec.num_classes == 10 for spec in DATASETS.values())
    assert DATASETS["mnist"].loader is mb.load_mnist
    with pytest.raises(ValueError):
        get_dataset("imagenet")


def test_check_tensors_accepts_valid_and_rejects_invalid_splits() -> None:
    spec = get_dataset("cifar10")
    x, y = torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 9])
    check_tensors(spec, x, y)
    for bad_x, bad_y in [
        (torch.rand(4, 1, 32, 32), y),             # wrong channels
        (torch.rand(4, 3, 28, 28), y),             # wrong size
        (x * 255, y),                              # not in [0, 1]
        (x, y.int()),                              # wrong label dtype
        (x, torch.tensor([0, 1, 2, 10])),          # label out of range
        (x, y[:3]),                                # length mismatch
    ]:
        with pytest.raises(ValueError):
            check_tensors(spec, bad_x, bad_y)


def test_load_dataset_checks_loader_output(monkeypatch: pytest.MonkeyPatch) -> None:
    good = (torch.rand(6, 3, 32, 32), torch.arange(6), torch.rand(2, 3, 32, 32), torch.arange(2))
    spec = DATASETS["cifar10"]
    monkeypatch.setitem(DATASETS, "cifar10", type(spec)(spec.name, spec.num_classes, spec.channels,
                                                        spec.image_size, lambda _: good))
    assert load_dataset("cifar10", "unused") is good
    bad = (good[0], good[1], torch.rand(2, 1, 32, 32), good[3])
    monkeypatch.setitem(DATASETS, "cifar10", type(spec)(spec.name, spec.num_classes, spec.channels,
                                                        spec.image_size, lambda _: bad))
    with pytest.raises(ValueError):
        load_dataset("cifar10", "unused")


def test_channel_stats_matches_per_channel_mean_and_std() -> None:
    x = torch.rand(50, 3, 8, 8)
    x[:, 1] *= 0.5
    mean, std = channel_stats(x)
    for c in range(3):
        assert mean[c].item() == pytest.approx(x[:, c].mean().item(), abs=1e-6)
        assert std[c].item() == pytest.approx(x[:, c].std().item(), abs=1e-5)


# -- dataset-aware entry points ---------------------------------------------------


def test_default_config_loads_as_mnist_through_runs() -> None:
    cfg = runs.load_config(mb.DEFAULT_CONFIG_PATH)
    assert cfg["dataset"] == "mnist"
    assert cfg == mb.load_config(mb.DEFAULT_CONFIG_PATH)


def test_mnist_validator_accepts_only_mnist_dataset_key() -> None:
    assert mb.validate_mnist_config({})["dataset"] == "mnist"
    assert mb.validate_mnist_config({"dataset": "mnist"})["dataset"] == "mnist"
    with pytest.raises(ValueError):
        mb.validate_mnist_config({"dataset": "cifar10"})


def test_dataset_of_reads_the_config_key(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("seeds: [0]\n", encoding="utf-8")
    assert runs.dataset_of(path) == "mnist"
    path.write_text("dataset: cifar10\nseeds: [0]\n", encoding="utf-8")
    assert runs.dataset_of(path) == "cifar10"
    with pytest.raises(FileNotFoundError):
        runs.dataset_of(tmp_path / "missing.yaml")


def test_pipeline_dispatch() -> None:
    assert runs.pipeline("mnist") is mb
    with pytest.raises(ValueError):
        runs.pipeline("imagenet")


def test_load_data_uses_the_configured_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(runs, "load_dataset", lambda name, data_dir: calls.append((name, data_dir)) or "data")
    assert runs.load_data({"dataset": "cifar10", "data_dir": "d"}) == "data"
    assert runs.load_data({"data_dir": "d"}) == "data"
    assert calls == [("cifar10", "d"), ("mnist", "d")]


# -- hooks in mnist_baseline (MNIST behaviour unchanged) ------------------------


def _legacy_salt_pepper(images: torch.Tensor, level: float, generator: torch.Generator) -> torch.Tensor:
    """The pre-change implementation: independent draws per value."""
    hit = torch.rand(images.shape, generator=generator) < level
    salt = torch.rand(images.shape, generator=generator) < 0.5
    out = images.clone()
    out[hit & salt] = 1.0
    out[hit & ~salt] = 0.0
    return out


def test_salt_pepper_is_unchanged_for_single_channel_images() -> None:
    x = torch.rand(7, 1, 28, 28)
    new = mb.corrupt_images(x, "salt_pepper", 0.2, torch.Generator().manual_seed(5))
    old = _legacy_salt_pepper(x, 0.2, torch.Generator().manual_seed(5))
    assert torch.equal(new, old)


def test_salt_pepper_hits_whole_pixels_on_colour_images() -> None:
    x = torch.full((20, 3, 32, 32), 0.5)
    out = mb.corrupt_images(x, "salt_pepper", 0.3, torch.Generator().manual_seed(1))
    changed = (out != x)
    assert torch.equal(changed.any(dim=1), changed.all(dim=1))         # all channels or none
    hit = changed.all(dim=1)
    assert torch.equal(out[:, 0][hit], out[:, 2][hit])                  # black or white, not coloured
    assert set(out[:, 0][hit].unique().tolist()) <= {0.0, 1.0}
    assert 0.25 < hit.float().mean().item() < 0.35


def test_occlusion_and_gaussian_work_on_colour_images() -> None:
    x = torch.rand(4, 3, 32, 32)
    occ = mb.corrupt_images(x, "occlusion", 8, torch.Generator().manual_seed(0))
    assert ((occ == 0).all(dim=1).flatten(1).sum(dim=1) >= 64).all()    # an 8x8 square per image
    g = mb.corrupt_images(x, "gaussian", 0.1, torch.Generator().manual_seed(0))
    assert g.shape == x.shape and g.min() >= 0 and g.max() <= 1


def _tiny_train(augment=None) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    net = mb.MnistNet(mb.MnistEncoder(c1=2, c2=2, hidden=4, embed_dim=4), torch.nn.Identity(),
                      mb.ClassifierHead(embed_dim=4))
    images = torch.rand(32, 1, 28, 28, generator=torch.Generator().manual_seed(1))
    labels = torch.arange(32) % 10
    mb.train_baseline(net, images, labels, epochs=1, batch_size=8, lr=1e-2, seed=3,
                      device=torch.device("cpu"), augment=augment)
    return {k: v.clone() for k, v in net.state_dict().items()}


def test_training_without_augment_is_deterministic_and_augment_is_applied() -> None:
    a, b = _tiny_train(), _tiny_train()
    assert all(torch.equal(a[k], b[k]) for k in a)
    seen = []

    def flip(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        seen.append((batch.shape[0], generator.initial_seed()))
        return batch.flip(-1)

    c = _tiny_train(augment=flip)
    assert len(seen) == 4 and all(s == (8, 3 + mb.AUGMENT_SEED_OFFSET) for s in seen)
    assert any(not torch.equal(a[k], c[k]) for k in a)
