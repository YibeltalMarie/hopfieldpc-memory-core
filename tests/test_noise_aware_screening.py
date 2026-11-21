"""Tests for Stage 3 (noise-aware facility location) and its shared additions: weighted greedy
facility location and the training-split column of memory_screening (synthetic; no data)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.memory_screening as ms
import benchmarks.mnist_baseline as mb
import benchmarks.noise_aware_screening as na
from benchmarks.memory_selection import coverage_similarity, greedy_facility_location, noise_aware_selection

C, PER, D = 10, 20, 6   # build_memory assumes 10 classes; PER >= max budget 16


def fake_run(seed: int = 0, n_val: int = 40) -> SimpleNamespace:
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


GRIDS = {"gaussian": [0.0, 0.1], "salt_pepper": [0.0, 0.05], "occlusion": [0, 2]}


# --- weighted greedy facility location (memory_selection) -------------------------------------


def test_weighted_greedy_matches_unweighted_with_unit_weights() -> None:
    q = torch.rand(30, 12, generator=torch.Generator().manual_seed(0)).double()
    plain, _ = greedy_facility_location(q, 5)
    weighted, _ = greedy_facility_location(q, 5, weights=torch.ones(30, dtype=torch.float64))
    assert torch.equal(plain, weighted)


def test_weights_change_what_is_covered_and_are_validated() -> None:
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    first, _ = greedy_facility_location(q, 1)                                   # two rows prefer column 1
    assert first.tolist() == [1]
    reweighted, _ = greedy_facility_location(q, 1, weights=torch.tensor([3.0, 1.0, 1.0], dtype=torch.float64))
    assert reweighted.tolist() == [0]
    with pytest.raises(ValueError):
        greedy_facility_location(q, 1, weights=torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        greedy_facility_location(q, 1, weights=torch.ones(2, dtype=torch.float64))


# --- noise-aware selection ----------------------------------------------------------------------


def test_gamma_zero_reduces_to_standard_facility_location() -> None:
    g = torch.Generator().manual_seed(1)
    z_clean, z_noisy = torch.randn(25, D, generator=g), torch.randn(25, D, generator=g)
    center = z_clean.mean(dim=0)
    idx, _ = greedy_facility_location(coverage_similarity(z_clean, z_clean, center), 6)
    picks = noise_aware_selection(z_clean, z_noisy, 6, 0.0, center, allow_noisy=False)
    assert torch.equal(picks, z_clean[idx])


def test_variant_b_can_store_a_corrupted_embedding_and_a_only_clean_ones() -> None:
    z_clean = torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]])
    z_noisy = torch.tensor([[0.0, 1.0], [0.1, 1.0], [-0.1, 1.0]])            # corrupted copies point elsewhere
    center = torch.zeros(2)
    b = noise_aware_selection(z_clean, z_noisy, 2, 1.0, center, allow_noisy=True)
    assert any(torch.equal(row, n) for row in b for n in z_noisy)
    a = noise_aware_selection(z_clean, z_noisy, 2, 1.0, center, allow_noisy=False)
    assert all(any(torch.equal(row, c) for c in z_clean) for row in a)


def test_design_names_are_complete_and_unique() -> None:
    names = na.design_names()
    assert len(names) == 1 + 3 * len(na.BUDGETS) + len(na.VARIANTS) * len(na.GAMMAS) * len(na.BUDGETS)
    assert len(set(names)) == len(names) and "na_fl_A_g0.25_k8" in names and "na_kmeans_k16" in names


def test_builder_shapes_and_variant_c_are_region_means() -> None:
    run = fake_run()
    designs = na.make_builder(GRIDS)(run, torch.device("cpu"))
    assert set(designs) == set(na.design_names()) - {"no_memory"}
    for name, d in designs.items():
        k = int(name.rsplit("_k", 1)[1])
        assert d["memory"].shape == (C * k, D)
    assert not torch.equal(designs["na_fl_C_g1_k4"]["memory"], designs["na_fl_A_g1_k4"]["memory"])
    reference = ms.build_designs(run, torch.device("cpu"))    # kmeans and FL identical to the earlier screening
    for k in na.BUDGETS:
        for base in ("kmeans", "facility_location"):
            assert torch.equal(designs[f"{base}_k{k}"]["memory"], reference[f"{base}_k{k}"]["memory"])


# --- training-split column (memory_screening) ---------------------------------------------------


def test_train_view_is_seeded_bounded_and_uses_its_own_noise_stream() -> None:
    run = fake_run()
    a, b = ms.TrainView(run, GRIDS, size=50), ms.TrainView(run, GRIDS, size=50)
    assert torch.equal(a.rows, b.rows) and a.rows.numel() == 50
    assert ms.TrainView(run, GRIDS).rows.numel() == C * PER                   # capped at the training size
    assert torch.equal(a.emb("gaussian", 0.0), a.z_val) and a.emb("gaussian", 0.1).shape == (50, D)
    assert ms.TRAIN_EVAL_STREAM not in (0, 1, 2)
    assert mb.corruption_seed(0, "gaussian", 0.1, stream=ms.TRAIN_EVAL_STREAM) != \
        mb.corruption_seed(0, "gaussian", 0.1, stream=2)


def test_stored_fraction() -> None:
    q = torch.eye(4)
    assert ms.stored_fraction(q, q[:2]) == pytest.approx(0.5)


def test_memory_gaps_subtract_the_no_memory_gap() -> None:
    raw = {"results": {"no_memory": [{"m": 0.80}, {"m": 0.82}], "x": [{"m": 0.85}, {"m": 0.86}]},
           "train_results": {"no_memory": [{"m": 0.90}, {"m": 0.92}], "x": [{"m": 0.97}, {"m": 0.98}]}}
    gap = ms.memory_gaps(raw, "m")["x"]
    assert gap["mean"] == pytest.approx(0.02) and gap["n"] == 2


# --- selection, decision, end to end ------------------------------------------------------------


def test_select_candidate_uses_tune_scores_and_prefers_smaller_gamma_on_ties() -> None:
    betas = [1.0, 10.0]
    curves = {}
    for v in na.VARIANTS:
        for g in na.GAMMAS:
            for k in na.BUDGETS:
                base = {"A": 0.5, "B": 0.6, "C": 0.4}[v]
                curves[na.na_name(v, g, k)] = [[base, base + (0.1 if g >= 0.5 else 0.0)]]
    sel = na.select_candidate({"beta_grid": betas, "beta_curves": curves})
    assert sel["candidate"] == "B"
    assert all(sel["gamma"][v][str(k)] == 0.5 for v in na.VARIANTS for k in na.BUDGETS)


def test_end_to_end_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "load_data", lambda cfg: None)
    monkeypatch.setattr(ms, "tune_backbone", lambda *a, **k: None)
    monkeypatch.setattr(ms, "prepare_run", lambda cfg, data, seed, device, retrain=False: fake_run(seed))
    cfg = {"dataset": "synthetic", "device": "cpu", "seeds": [0, 1], "beta_values": [1.0, 10.0],
           "beta_selection_sets": [("clean", 0.0), ("gaussian", 0.05)],
           "gaussian_sigmas": [0.0, 0.1], "salt_pepper_probs": [0.0, 0.05], "occlusion_sizes": [0, 2]}
    raw = na.run_noise_aware(cfg)
    assert set(raw["results"]) == set(na.design_names()) == set(raw["train_results"])
    assert raw["decision"]["band"] in ("Pass", "Weak positive", "Fail")
    assert raw["decision"]["candidate"] == raw["selection"]["candidate"]
    text = na.build_report(raw)
    for section in ("## 1. Tune-half selection", "## 2. Decision", "attribution", "## 4. Training / validation",
                    "occlusion AUC"):
        assert section in text
