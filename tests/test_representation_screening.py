"""Tests for Stage 2 (selection x representation) and the shared Phase 1 bands (synthetic; no data)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.memory_screening as ms
import benchmarks.mnist_baseline as mb
import benchmarks.representation_screening as rs
from benchmarks.memory_diagnostics import region_means as cosine_region_means
from benchmarks.screening_criteria import method_band, band_at_budget

C, PER, D = 10, 20, 4


def fake_run(seed: int = 0, n_val: int = 40) -> SimpleNamespace:
    g = torch.Generator().manual_seed(seed)
    centers = 3.0 * torch.randn(C, D, generator=g)
    y = torch.arange(C).repeat_interleave(PER)
    z = centers[y] + torch.randn(C * PER, D, generator=g)
    yv = torch.arange(n_val) % C
    zv = centers[yv] + torch.randn(n_val, D, generator=g)
    noise = torch.randn(n_val, D, generator=g)
    torch.manual_seed(seed)
    net = mb.MnistNet(nn.Identity(), nn.Identity(), mb.ClassifierHead(embed_dim=D))
    with torch.no_grad():
        net.head.fc.weight.copy_(centers)
        net.head.fc.bias.copy_(-0.5 * (centers ** 2).sum(dim=1))
    run = SimpleNamespace(seed=seed, z_train=z, z_val=zv, splits=SimpleNamespace(train_labels=y, val_labels=yv),
                          batch_size=16, device=torch.device("cpu"), net=net, from_cache=True)
    run.val_embeddings = lambda kind, level: zv if kind == "clean" or level == 0 else zv + 4 * level * noise
    return run


def test_design_names_cover_every_base_and_budget() -> None:
    names = rs.design_names()
    assert names[0] == "no_memory" and len(names) == 1 + len(rs.BASES) * len(rs.BUDGETS)
    assert len(set(names)) == len(names) and "facility_location_rm_k8" in names


def test_region_means_equal_cluster_means_for_separated_clusters() -> None:
    a = torch.tensor([[10.0, 0.0], [10.0, 1.0], [10.0, -1.0]])
    b = torch.tensor([[0.0, 10.0], [1.0, 10.0], [-1.0, 10.0]])
    z, y = torch.cat([a, b]), torch.zeros(6, dtype=torch.long)
    picks, pick_labels = torch.stack([a[1], b[2]]), torch.zeros(2, dtype=torch.long)
    expected = torch.stack([a.mean(dim=0), b.mean(dim=0)])
    assert torch.allclose(rs.euclidean_region_means(z, y, picks, pick_labels), expected)
    means, _ = cosine_region_means(z, y, picks, pick_labels, None)
    assert torch.allclose(means, expected)


def test_cosine_and_euclidean_assignment_can_differ() -> None:
    # (9, 0) points the same way as (1, 0) but lies next to (10, 1).
    z = torch.tensor([[1.0, 0.0], [10.0, 1.0], [9.0, 0.0]])
    y = torch.zeros(3, dtype=torch.long)
    picks, pick_labels = z[:2].clone(), torch.zeros(2, dtype=torch.long)
    cos_means, _ = cosine_region_means(z, y, picks, pick_labels, None)
    euc_means = rs.euclidean_region_means(z, y, picks, pick_labels)
    assert torch.allclose(cos_means[0], torch.tensor([5.0, 0.0]))       # (1,0) and (9,0) by angle
    assert torch.allclose(euc_means[1], torch.tensor([9.5, 0.5]))       # (10,1) and (9,0) by distance


def test_build_designs_shapes_reality_and_averaging() -> None:
    run = fake_run()
    designs = rs.build_designs(run, torch.device("cpu"))
    assert set(designs) == set(rs.design_names()) - {"no_memory"}
    for k in rs.BUDGETS:
        for base in rs.BASES:
            d = designs[f"{base}_k{k}"]
            assert d["memory"].shape == (C * k, D) and torch.equal(d["labels"], torch.arange(C).repeat_interleave(k))
    k_means = ms.build_designs(run, torch.device("cpu"))   # Stage 2 reuses the screening's memories exactly
    for k in rs.BUDGETS:
        for base in ("kmeans", "facility_location", "random"):
            assert torch.equal(designs[f"{base}_k{k}"]["memory"], k_means[f"{base}_k{k}"]["memory"])
    for row in designs["kmeans_medoid_k8"]["memory"]:      # medoids are real training embeddings
        assert (run.z_train - row).abs().sum(dim=1).min() < 1e-6
    assert not torch.equal(designs["facility_location_rm_k8"]["memory"], designs["facility_location_k8"]["memory"])


def summary(mean: float, n_pos: int, n: int = 10) -> dict:
    return {"mean": mean, "n_pos": n_pos, "n": n}


def test_band_at_budget_thresholds() -> None:
    clean_ok = summary(0.0, 5)
    assert band_at_budget(summary(0.004, 8), clean_ok) == "pass"
    assert band_at_budget(summary(0.003, 7), clean_ok) == "pass"
    assert band_at_budget(summary(0.002, 9), clean_ok) == "weak"
    assert band_at_budget(summary(0.0005, 9), clean_ok) == "fail"
    assert band_at_budget(summary(0.004, 6), clean_ok) == "fail"                  # too few positive seeds
    assert band_at_budget(summary(0.004, 9), summary(-0.004, 1)) == "fail"        # clean loss > 0.3 pp


def test_method_band_needs_two_budgets() -> None:
    assert method_band({2: "pass", 4: "pass", 8: "fail", 16: "fail"}) == "Pass"
    assert method_band({2: "pass", 4: "weak", 8: "fail", 16: "fail"}) == "Weak positive"
    assert method_band({2: "pass", 4: "fail", 8: "fail", 16: "fail"}) == "Fail"


def test_run_end_to_end_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "load_data", lambda cfg: None)
    monkeypatch.setattr(ms, "tune_backbone", lambda *a, **k: None)
    monkeypatch.setattr(ms, "prepare_run", lambda cfg, data, seed, device, retrain=False: fake_run(seed))
    cfg = {"dataset": "synthetic", "device": "cpu", "seeds": [0, 1], "beta_values": [1.0, 10.0],
           "beta_selection_sets": [("clean", 0.0), ("gaussian", 0.05)],
           "gaussian_sigmas": [0.0, 0.1], "salt_pepper_probs": [0.0, 0.05], "occlusion_sizes": [0, 8]}
    raw = rs.run_representation(cfg)
    assert set(raw["results"]) == set(rs.design_names())
    dec = rs.candidate_decision(raw)
    assert dec["band"] in ("Pass", "Weak positive", "Fail") and set(dec["per_budget"]) == {"2", "4", "8", "16"}
    text = rs.build_report(raw)
    assert "## 1. Decision" in text and "Euclidean vs cosine" in text and "kmeans_medoid_k16" in text
