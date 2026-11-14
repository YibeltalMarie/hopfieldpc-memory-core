"""Tests for the CIFAR-10 model, pipeline, config and pilot (synthetic data, no download)."""

from __future__ import annotations

import pytest
import torch

import benchmarks.mnist_baseline as mb
from benchmarks import cifar10_baseline as cb
from benchmarks import runs
from benchmarks.cifar10_model import (
    CROP_PADDING,
    augment_batch,
    build_cifar10_net,
    validate_model_config,
)
from benchmarks.cifar10_pilot import parse_candidates, pilot_candidate

TINY = {"name": "cnn", "widths": [4, 4], "embed_dim": 4}
MEAN, STD = torch.tensor([0.5, 0.4, 0.3]), torch.tensor([0.2, 0.2, 0.25])


def synthetic_data(n_train: int = 80, n_test: int = 20):
    g = torch.Generator().manual_seed(0)
    return (torch.rand(n_train, 3, 32, 32, generator=g), torch.arange(n_train) % 10,
            torch.rand(n_test, 3, 32, 32, generator=g), torch.arange(n_test) % 10)


def tiny_cfg(tmp_path, **overrides):
    base = {"dataset": "cifar10", "seeds": [0], "reports_dir": str(tmp_path), "val_size": 20,
            "epochs": 1, "batch_size": 16, "lr": 1e-2, "eval_batch_size": 32, "model": TINY,
            "gaussian_sigmas": [0.0, 0.1], "salt_pepper_probs": [0.0, 0.05], "occlusion_sizes": [0, 8],
            "beta_selection_sets": [["clean", 0], ["gaussian", 0.12]]}
    return cb.validate_cifar10_config({**base, **overrides})


# -- model ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [TINY, {"name": "resnet18", "embed_dim": 7}])
def test_networks_map_images_to_embeddings_and_logits(model) -> None:
    net = build_cifar10_net(0, model, MEAN, STD).eval()
    logits, z, h, aux = net(torch.rand(2, 3, 32, 32))
    assert z.shape == (2, model["embed_dim"]) and logits.shape == (2, 10) and torch.equal(z, h)
    assert isinstance(net.bridge, torch.nn.Identity)


def test_normalization_buffers_are_saved_and_used() -> None:
    net = build_cifar10_net(0, TINY, MEAN, STD).eval()
    state = net.state_dict()
    assert torch.equal(state["encoder.mean"].flatten(), MEAN) and torch.equal(state["encoder.std"].flatten(), STD)
    other = build_cifar10_net(0, TINY, torch.zeros(3), torch.ones(3)).eval()
    x = torch.rand(3, 3, 32, 32)
    assert not torch.allclose(other.encoder(x), net.encoder(x))
    other.load_state_dict(state)
    assert torch.equal(other.encoder(x), net.encoder(x))


def test_seeded_initialization_is_reproducible() -> None:
    a, b = build_cifar10_net(3, TINY, MEAN, STD), build_cifar10_net(3, TINY, MEAN, STD)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_validate_model_config() -> None:
    assert validate_model_config(None)["name"] == "cnn"
    assert validate_model_config({"name": "resnet18"}) == {"name": "resnet18", "embed_dim": 64}
    for bad in [{"name": "vgg"}, {"name": "resnet18", "widths": [8]}, {"name": "cnn", "widths": [8] * 6},
                {"name": "cnn", "embed_dim": True}, {"name": "cnn", "widths": []}, "cnn"]:
        with pytest.raises(ValueError):
            validate_model_config(bad)


def test_augment_batch_is_a_seeded_crop_or_flipped_crop() -> None:
    x = torch.rand(6, 3, 32, 32)
    a = augment_batch(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape
    assert torch.equal(a, augment_batch(x, torch.Generator().manual_seed(1)))
    padded = torch.nn.functional.pad(x, (CROP_PADDING,) * 4)
    for i in range(6):
        crops = [padded[i, :, t:t + 32, s:s + 32] for t in range(9) for s in range(9)]
        assert any(torch.equal(a[i], c) or torch.equal(a[i], c.flip(-1)) for c in crops)


# -- config ----------------------------------------------------------------------------


def test_default_and_quick_configs_load(tmp_path) -> None:
    cfg = cb.load_config(cb.DEFAULT_CONFIG_PATH)
    assert cfg["dataset"] == "cifar10" and cfg["model"]["name"] == "cnn" and cfg["augment"] is True
    assert cfg["placements"] == ["embedding"] and cfg["baseline_tuning"] is None
    quick = cb.load_config(cb.DEFAULT_CONFIG_PATH, quick=True)
    assert quick["seeds"] == [0] and quick["train_subset"] == 2000 and quick["quick"] is True
    assert quick["cost"]["batch_size"] == cfg["cost"]["batch_size"]   # cost merges per key
    assert runs.load_config(cb.DEFAULT_CONFIG_PATH) == cfg
    assert runs.pipeline("cifar10") is cb


def test_cifar10_config_rejects_invalid_settings(tmp_path) -> None:
    for bad in [{"dataset": "mnist"}, {"augment": "yes"}, {"occlusion_sizes": [0, 40]},
                {"placements": ["pixel"]}, {"unknown_key": 1},
                {"baseline_tuning": {"seeds": [0], "lr_values": [0.001], "batch_size_values": [64], "max_epochs": 2}}]:
        with pytest.raises((ValueError, TypeError)):
            tiny_cfg(tmp_path, **bad)


def test_fingerprint_tracks_model_and_augmentation(tmp_path) -> None:
    a = cb.checkpoint_fingerprint(tiny_cfg(tmp_path), 0, 60)
    assert a != cb.checkpoint_fingerprint(tiny_cfg(tmp_path, augment=False), 0, 60)
    assert a != cb.checkpoint_fingerprint(tiny_cfg(tmp_path, model={**TINY, "embed_dim": 5}), 0, 60)


# -- pipeline ----------------------------------------------------------------------------


def test_backbone_is_cached_and_retrained_when_settings_change(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    splits = mb.make_splits(synthetic_data(), seed=0, val_size=cfg["val_size"])
    net1, _, cached1 = cb.get_or_train_baseline(cfg, splits, 0, torch.device("cpu"))
    net2, _, cached2 = cb.get_or_train_baseline(cfg, splits, 0, torch.device("cpu"))
    assert (cached1, cached2) == (False, True)
    assert all(torch.equal(net1.state_dict()[k], net2.state_dict()[k]) for k in net1.state_dict())
    _, _, cached3 = cb.get_or_train_baseline(tiny_cfg(tmp_path, lr=5e-3), splits, 0, torch.device("cpu"))
    assert cached3 is False


def test_normalization_comes_from_the_training_split_only(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    data = synthetic_data()
    splits = mb.make_splits(data, seed=0, val_size=cfg["val_size"])
    net, _, _ = cb.get_or_train_baseline(cfg, splits, 0, torch.device("cpu"))
    train_mean = splits.train_images.double().mean(dim=(0, 2, 3)).float()
    assert torch.allclose(net.encoder.mean.flatten(), train_mean, atol=1e-6)


def test_prepare_baseline_run_builds_a_standard_run(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    run = runs.prepare_run(cfg, synthetic_data(), 0, torch.device("cpu"))
    assert isinstance(run, mb.BaselineRun)
    assert run.z_train.shape == (60, 4) and run.z_val.shape == (20, 4)
    assert run.prototypes.shape == (10, 4) and torch.equal(run.memory_labels, torch.arange(10))
    assert run.val_embeddings("salt_pepper", 0.05).shape == (20, 4)
    assert 0.0 <= cb.val_head_accuracy(run, "clean", 0.0) <= 1.0
    assert cb.tune_baseline(cfg, None, torch.device("cpu")) is None


# -- pilot -------------------------------------------------------------------------------------


def test_parse_candidates() -> None:
    assert parse_candidates(["cnn_small:30", "resnet18:1"]) == [("cnn_small", 30), ("resnet18", 1)]
    for bad in (["cnn_small"], ["vgg:3"], ["cnn_small:0"]):
        with pytest.raises(ValueError):
            parse_candidates(bad)


def test_pilot_candidate_logs_every_epoch(tmp_path) -> None:
    cfg = tiny_cfg(tmp_path)
    splits = mb.make_splits(synthetic_data(), seed=0, val_size=cfg["val_size"])
    result = pilot_candidate(cfg, splits, TINY, 2, 0, torch.device("cpu"))
    assert [e["epoch"] for e in result["log"]] == [1, 2]
    assert all(0 <= e["val_clean"] <= 1 and e["train_s"] >= 0 for e in result["log"])
    assert result["best_epoch"] in (1, 2) and result["params"] > 0
