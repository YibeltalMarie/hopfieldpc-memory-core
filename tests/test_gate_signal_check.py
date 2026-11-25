"""Tests for the gate-signal check (synthetic; no data)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.gate_signal_check as gs
import benchmarks.key_value_screening as kv
import benchmarks.memory_screening as ms
import benchmarks.mnist_baseline as mb

C, PER, D = 10, 20, 6   # build_memory assumes 10 classes; PER >= max budget 16
GRIDS = {"gaussian": [0.0, 0.1], "salt_pepper": [0.0, 0.05], "occlusion": [0, 2]}
CFG = {"dataset": "synthetic", "device": "cpu", "seeds": [0, 1], "beta_values": [1.0, 10.0],
       "beta_selection_sets": [("clean", 0.0), ("gaussian", 0.05)],
       "gaussian_sigmas": [0.0, 0.1], "salt_pepper_probs": [0.0, 0.05], "occlusion_sizes": [0, 2]}


def fake_run(seed: int = 0, n_val: int = 40) -> SimpleNamespace:
    """Synthetic seed run (same construction as tests/test_noise_aware_screening.py)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(C).repeat_interleave(PER)
    pattern = torch.eye(C, 16)[y].view(-1, 1, 4, 4)
    images = (0.5 + 0.4 * pattern + 0.02 * torch.randn(C * PER, 1, 4, 4, generator=g)).clamp(0, 1)
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(16, D))
    head = mb.ClassifierHead(embed_dim=D)
    with torch.no_grad():
        z = encoder(images)
        means = torch.stack([z[y == c].mean(dim=0) for c in range(C)])
        head.fc.weight.copy_(means)
        head.fc.bias.copy_(-0.5 * (means ** 2).sum(dim=1))
    net = mb.MnistNet(encoder, nn.Identity(), head)
    yv = torch.arange(n_val) % C
    zv = z[yv * PER + 1] + 0.05 * torch.randn(n_val, D, generator=g)
    noise = torch.randn(n_val, D, generator=g)
    run = SimpleNamespace(seed=seed, z_train=z, z_val=zv, net=net, batch_size=32, device=torch.device("cpu"),
                          from_cache=True, splits=SimpleNamespace(train_labels=y, val_labels=yv, train_images=images))
    run.val_embeddings = lambda kind, level: zv if kind == "clean" or level == 0 else zv + 4 * level * noise
    return run


def patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "load_data", lambda cfg: None)
    monkeypatch.setattr(ms, "tune_backbone", lambda *a, **k: None)
    monkeypatch.setattr(ms, "prepare_run", lambda cfg, data, seed, device, retrain=False: fake_run(seed))


def test_auroc_perfect_none_and_ties() -> None:
    assert gs.auroc(torch.tensor([0.0, 0.1]), torch.tensor([0.5, 0.9])) == 1.0
    assert gs.auroc(torch.tensor([0.5, 0.9]), torch.tensor([0.0, 0.1])) == 0.0
    assert gs.auroc(torch.tensor([0.3, 0.3]), torch.tensor([0.3, 0.3])) == 0.5
    assert abs(gs.auroc(torch.tensor([0.0, 0.4]), torch.tensor([0.2, 0.6])) - 0.75) < 1e-12


def test_bins_and_set_summary() -> None:
    g = torch.tensor([0.0, 0.25, 0.6, 1.0])
    assert gs.bin_index(g).tolist() == [0, 1, 2, 3]
    ok_ref, ok_kv = torch.tensor([True, True, False, False]), torch.tensor([True, False, True, True])
    s = gs.set_summary(g, ok_ref, ok_kv, torch.tensor([0.0, 0.1]))
    assert s["above"]["0.5"] == 0.5 and s["bins"][2] == {"share": 0.25, "acc_ref": 0.0, "acc_kv": 1.0}
    assert s["acc_ref"] == 0.5 and s["acc_kv"] == 0.75 and 0.0 <= s["auroc_vs_clean"] <= 1.0


def test_end_to_end_reproduces_stage4(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_pipeline(monkeypatch)
    stage4 = kv.run_key_value(CFG)
    raw = gs.run_check(CFG, stage4)
    assert raw["reproduction_gap"] < 1e-12
    text = gs.build_report(raw)
    assert "## 1. Distribution of g" in text and "## 2. Accuracy by bin" in text
