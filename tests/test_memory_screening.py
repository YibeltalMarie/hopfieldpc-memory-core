"""Tests for the Phase 1 memory-construction screening (synthetic runs; no data or training)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.memory_screening as ms
import benchmarks.mnist_baseline as mb

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
    with torch.no_grad():   # a head that classifies by the class centres
        net.head.fc.weight.copy_(centers)
        net.head.fc.bias.copy_(-0.5 * (centers ** 2).sum(dim=1))
    run = SimpleNamespace(seed=seed, z_train=z, z_val=zv, splits=SimpleNamespace(train_labels=y, val_labels=yv),
                          batch_size=16, device=torch.device("cpu"), net=net, from_cache=True)
    run.val_embeddings = lambda kind, level: zv if kind == "clean" or level == 0 else zv + 4 * level * noise
    return run


def test_design_names() -> None:
    names = ms.design_names()
    assert names[:2] == ["no_memory", "class_means"] and names[-1] == "all_train"
    assert len(names) == 2 + 3 * len(ms.BUDGETS) + 1 and len(set(names)) == len(names)
    assert ms.REFERENCE_DESIGN in names


def test_build_designs_sizes_centres_and_nesting() -> None:
    run = fake_run()
    designs = ms.build_designs(run, torch.device("cpu"))
    assert set(designs) == set(ms.design_names()) - {"no_memory"}
    for name, d in designs.items():
        assert torch.allclose(d["center"], run.z_train.mean(dim=0))
        assert d["memory"].shape[0] == d["labels"].shape[0]
    assert designs["class_means"]["memory"].shape == (C, D)
    assert designs["kmeans_k16"]["memory"].shape == (C * 16, D)
    assert torch.equal(designs["all_train"]["memory"], run.z_train)
    fl2, fl16 = designs["facility_location_k2"]["memory"], designs["facility_location_k16"]["memory"]
    for c in range(C):   # nested prefixes of the k = 16 selection
        assert torch.equal(fl2[2 * c:2 * c + 2], fl16[16 * c:16 * c + 2])
    for row in designs["random_k8"]["memory"]:   # real training embeddings
        assert (run.z_train - row).abs().sum(dim=1).min() < 1e-6


def test_valview_returns_clean_and_cached_noisy_embeddings() -> None:
    run = fake_run()
    view = ms.ValView(run, [("clean", 0.0), ("gaussian", 0.1), ("occlusion", 0.0)])
    assert view.emb("clean", 0) is run.z_val and view.emb("occlusion", 0) is run.z_val
    assert torch.equal(view.emb("gaussian", 0.1), run.val_embeddings("gaussian", 0.1))
    with pytest.raises(KeyError):
        view.emb("gaussian", 0.2)


def test_accuracy_matches_a_direct_count() -> None:
    run = fake_run()
    expected = (run.net.head(run.z_val).argmax(dim=-1) == run.splits.val_labels).float().mean().item()
    assert ms.accuracy(run.net, run.z_val, run.splits.val_labels, 7, run.device) == pytest.approx(expected)


def test_evaluate_reports_clean_levels_aucs_and_their_mean() -> None:
    run = fake_run()
    grids = {"gaussian": [0.0, 0.1, 0.2], "salt_pepper": [0.0, 0.05], "occlusion": [0.0, 8.0]}
    view = ms.ValView(run, [(k, lv) for k, lvs in grids.items() for lv in lvs])
    idx = torch.arange(run.z_val.shape[0])
    out = ms.evaluate(run.net, view, idx, grids)
    assert out["gaussian_acc"][0] == out["clean"] and len(out["gaussian_acc"]) == 3
    for k, lvs in grids.items():
        assert out[f"{k}_auc"] == pytest.approx(mb.robust_auc(lvs, out[f"{k}_acc"]))
    assert out["noise_auc"] == pytest.approx(sum(out[f"{k}_auc"] for k in grids) / 3)


def test_headroom_decision_bands() -> None:
    assert ms.headroom_decision(1.2).startswith("GO:")
    assert ms.headroom_decision(1.0).startswith("GO:")
    assert ms.headroom_decision(0.7).startswith("GO, modest")
    assert ms.headroom_decision(0.2).startswith("REPORT")


def synthetic_raw() -> dict:
    seeds = list(range(8))
    offset = {n: 0.0 for n in ms.design_names()}
    offset.update({"all_train": 0.02, "kmeans_k16": 0.004, "class_means": -0.01})
    rows = {n: [{"clean": 0.9 + offset[n] / 4, "gaussian_auc": 0.5 + offset[n] + 0.001 * s,
                 "salt_pepper_auc": 0.5 + offset[n], "occlusion_auc": 0.7 + offset[n],
                 "noise_auc": (1.7 + 3 * offset[n] + 0.001 * s) / 3} for s in seeds] for n in ms.design_names()}
    return {"dataset": "cifar10", "seeds": seeds, "budgets": list(ms.BUDGETS), "results": rows,
            "grids": {"gaussian": [0, 0.1], "salt_pepper": [0, 0.05], "occlusion": [0, 8]},
            "beta_grid": [1.0, 10.0, 10000.0], "beta_star": {n: 10.0 for n in ms.design_names() if n != "no_memory"},
            "beta_selection_sets": [["clean", 0]],
            "memory_size": {n: 1 for n in ms.design_names() if n != "no_memory"},
            "build_s": {n: [0.1] for n in ms.design_names() if n != "no_memory"}}


def test_comparisons_are_paired_against_kmeans_k8() -> None:
    comp = ms.comparisons(synthetic_raw())
    assert ms.REFERENCE_DESIGN not in comp
    assert comp["all_train"]["mean"] == pytest.approx(0.02) and comp["all_train"]["n_pos"] == 8
    assert comp["class_means"]["mean"] == pytest.approx(-0.01)


def test_report_states_the_headroom_decision() -> None:
    text = ms.build_report(synthetic_raw())
    assert "## 3. Headroom" in text and "+2.00 pp" in text
    assert "GO: clear headroom" in text
    assert "## 4. Facility location" in text and "| 16 |" in text


def test_run_screening_end_to_end_on_synthetic_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "load_data", lambda cfg: None)
    monkeypatch.setattr(ms, "tune_backbone", lambda *a, **k: None)
    monkeypatch.setattr(ms, "prepare_run", lambda cfg, data, seed, device, retrain=False: fake_run(seed))
    cfg = {"dataset": "synthetic", "device": "cpu", "seeds": [0, 1], "beta_values": [1.0, 10.0],
           "beta_selection_sets": [("clean", 0.0), ("gaussian", 0.05)],
           "gaussian_sigmas": [0.0, 0.1], "salt_pepper_probs": [0.0, 0.05], "occlusion_sizes": [0, 8]}
    raw = ms.run_screening(cfg)
    assert set(raw["results"]) == set(ms.design_names())
    assert all(len(rows) == 2 for rows in raw["results"].values())
    assert raw["beta_grid"] == [1.0, 10.0, *ms.EXTRA_BETAS]
    assert raw["memory_size"]["all_train"] == C * PER and raw["memory_size"]["kmeans_k8"] == C * 8
    assert all(0.0 <= r["noise_auc"] <= 1.0 for rows in raw["results"].values() for r in rows)
    assert "Headroom" in ms.build_report(raw)
