"""Tests for the Stage 4 follow-up retrieval-weight diagnostic (synthetic; no data)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.key_value_diagnostics as kd
import benchmarks.key_value_screening as kv
import benchmarks.memory_screening as ms
import benchmarks.mnist_baseline as mb
from benchmarks.mnist_prototype_retrieval import HopfieldBridge

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


def test_designs_match_the_stage4_builder() -> None:
    run = fake_run()
    mine = kd.build_designs(run, GRIDS, torch.device("cpu"))
    stage4 = kv.make_builder(GRIDS)(run, torch.device("cpu"))
    for n in kd.DESIGNS:
        assert torch.equal(mine[n]["memory"], stage4[n]["memory"])
        assert torch.equal(mine[n]["labels"], stage4[n]["labels"])
        assert ("values" in mine[n]) == ("values" in stage4[n])
        if "values" in mine[n]:
            assert torch.equal(mine[n]["values"], stage4[n]["values"])
    assert not mine[kd.REFERENCE]["noisy"].any()
    assert abs(mine[kd.M_NAME]["noisy"].double().mean().item() - stage4[kd.M_NAME]["info"]["noisy_keys"]) < 1e-9


def test_query_stats_weight_breakdown_by_hand() -> None:
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    labels = torch.tensor([0, 1, 1])
    noisy = torch.tensor([False, True, False])
    bridge = HopfieldBridge(keys, labels, beta=1.0, center=None)
    head = mb.ClassifierHead(embed_dim=2)
    net = mb.MnistNet(nn.Identity(), bridge, head)
    z, y = torch.tensor([[1.0, 0.2], [0.1, 1.0]]), torch.tensor([0, 0])
    stats = kd.query_stats(net, z, y, noisy, labels, 8, torch.device("cpu"))
    w = bridge(z)[1]["weights"].double()
    assert torch.allclose(stats["mass_noisy"], w[:, 1])
    assert torch.allclose(stats["mass_wrong"], w[:, 1] + w[:, 2])
    assert torch.allclose(stats["mass_wrong_noisy"], w[:, 1])
    assert stats["top_wrong"].tolist() == [False, True] and stats["top_noisy"].tolist() == [False, True]
    assert kd.category(stats).tolist() == [kd.CATEGORIES.index("right_clean"), kd.CATEGORIES.index("wrong_noisy")]


def test_lost_queries_counts_and_categories() -> None:
    ref = {"correct": torch.tensor([True, True, False, True])}
    new = {"correct": torch.tensor([False, True, True, False]),
           "top_wrong": torch.tensor([True, False, False, False]),
           "top_noisy": torch.tensor([True, False, False, True])}
    out = kd.lost_queries(ref, new)
    assert out["lost"] == 0.5 and out["gained"] == 0.25
    assert out["lost_top"]["wrong_noisy"] == 0.5 and out["lost_top"]["right_noisy"] == 0.5


def test_end_to_end_reproduces_stage4_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_pipeline(monkeypatch)
    stage4 = kv.run_key_value(CFG)
    raw = kd.run_diagnostics(CFG, stage4)
    assert raw["reproduction_gap"] < 1e-12
    text = kd.build_report(raw)
    for section in ("## 1. Accuracy", "## 2.", "## 3. Queries lost", "Reproduction check"):
        assert section in text
