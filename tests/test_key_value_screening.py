"""Tests for Stage 4 (key-value noise-aware facility location): selection, served means, the
key-value bridge and the screening module (synthetic; no data)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.key_value_screening as kv
import benchmarks.memory_screening as ms
import benchmarks.mnist_baseline as mb
import benchmarks.noise_aware_screening as na
from benchmarks.memory_selection import key_value_selection, noise_aware_selection, served_clean_means
from benchmarks.mnist_prototype_retrieval import HopfieldBridge

C, PER, D = 10, 20, 6   # build_memory assumes 10 classes; PER >= max budget 16
GRIDS = {"gaussian": [0.0, 0.1], "salt_pepper": [0.0, 0.05], "occlusion": [0, 2]}


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


# --- selection ----------------------------------------------------------------------------------


def test_keys_are_variant_b_picks_and_values_their_clean_sources() -> None:
    g = torch.Generator().manual_seed(2)
    z_clean = torch.randn(30, D, generator=g)
    z_noisy = z_clean + 1.5 * torch.randn(30, D, generator=g)
    center = z_clean.mean(dim=0)
    keys, values, is_noisy = key_value_selection(z_clean, z_noisy, 8, 1.0, center)
    assert torch.equal(keys, noise_aware_selection(z_clean, z_noisy, 8, 1.0, center, allow_noisy=True))
    for key, value, noisy in zip(keys, values, is_noisy):
        rows = z_noisy if noisy else z_clean
        i = next(j for j in range(30) if torch.equal(rows[j], key))
        assert torch.equal(value, z_clean[i])


def test_a_corrupted_key_returns_the_clean_embedding_of_its_source() -> None:
    z_clean = torch.tensor([[1.0, 0.0], [1.0, 0.1], [1.0, -0.1]])
    z_noisy = torch.tensor([[0.0, 1.0], [0.1, 1.0], [-0.1, 1.0]])            # corrupted copies point elsewhere
    keys, values, is_noisy = key_value_selection(z_clean, z_noisy, 2, 1.0, torch.zeros(2))
    assert is_noisy.any() and not is_noisy.all()
    j = int(is_noisy.nonzero()[0])
    i = next(r for r in range(3) if torch.equal(z_noisy[r], keys[j]))
    assert torch.equal(values[j], z_clean[i])
    with pytest.raises(ValueError):
        key_value_selection(z_clean, z_noisy[:2], 2, 1.0, None)


def test_served_clean_means_weights_and_fallback() -> None:
    z_clean = torch.tensor([[1.0, 0.0], [1.0, 0.2], [0.0, 1.0]])
    z_noisy = torch.tensor([[0.9, 0.1], [0.0, 1.0], [0.1, 1.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])             # third key serves nothing
    fallback = torch.full((3, 2), 7.0)
    out = served_clean_means(z_clean, z_noisy, keys, 0.5, None, fallback=fallback)
    # key 0 serves clean 0, clean 1 (weight 1) and noisy 0 (weight 0.5; source clean 0)
    expected0 = (z_clean[0] + z_clean[1] + 0.5 * z_clean[0]) / 2.5
    # key 1 serves clean 2 (1) and noisy 1, noisy 2 (0.5 each; sources clean 1 and clean 2)
    expected1 = (z_clean[2] + 0.5 * z_clean[1] + 0.5 * z_clean[2]) / 2.0
    assert torch.allclose(out[0], expected0) and torch.allclose(out[1], expected1)
    assert torch.equal(out[2], fallback[2])


# --- key-value bridge ---------------------------------------------------------------------------


def test_bridge_with_values_equal_to_keys_matches_hopfield_bridge() -> None:
    g = torch.Generator().manual_seed(3)
    keys, labels = torch.randn(12, D, generator=g), torch.arange(12) % 4
    center, z = torch.randn(D, generator=g), torch.randn(5, D, generator=g)
    plain = HopfieldBridge(keys, labels, beta=8.0, center=center)
    same = ms.KeyValueBridge(keys, keys, labels, beta=8.0, center=center)
    assert torch.allclose(same(z)[0], plain(z)[0], atol=1e-5)


def test_bridge_returns_values_weighted_by_key_scores() -> None:
    g = torch.Generator().manual_seed(4)
    keys, values, labels = torch.randn(6, D, generator=g), torch.randn(6, D, generator=g), torch.arange(6) % 2
    bridge = ms.KeyValueBridge(keys, values, labels, beta=5.0, center=keys.mean(dim=0))
    z = torch.randn(3, D, generator=g)
    h, aux = bridge(z)
    reference_weights = HopfieldBridge(keys, labels, beta=5.0, center=keys.mean(dim=0))(z)[1]["weights"]
    assert torch.allclose(aux["weights"], reference_weights)
    assert torch.allclose(h, aux["weights"] @ values, atol=1e-6)
    bridge.set_retrieval(beta=50.0)
    assert bridge.beta == 50.0
    with pytest.raises(ValueError):
        bridge.set_retrieval(num_updates=2)
    with pytest.raises(ValueError):
        ms.KeyValueBridge(keys, values[:5], labels, beta=1.0)


def test_make_net_uses_the_key_value_bridge_only_when_values_are_given() -> None:
    run = fake_run()
    design = {"memory": run.z_train[:C], "labels": torch.arange(C), "center": run.z_train.mean(dim=0)}
    assert type(ms.make_net(run, design).bridge) is HopfieldBridge
    assert isinstance(ms.make_net(run, {**design, "values": run.z_train[C:2 * C]}).bridge, ms.KeyValueBridge)


# --- screening module ---------------------------------------------------------------------------


def test_design_names_are_complete_and_unique() -> None:
    names = kv.design_names()
    n_grid = len(kv.GAMMAS) * len(kv.BUDGETS)
    assert len(names) == 1 + 3 * len(kv.BUDGETS) + 2 * n_grid + len(kv.KV_VARIANTS) * n_grid
    assert len(set(names)) == len(names) and "kv_S_g2_k8" in names and "na_fl_B_g0.25_k16" in names


def test_builder_keys_values_and_shared_designs() -> None:
    run = fake_run()
    designs = kv.make_builder(GRIDS)(run, torch.device("cpu"))
    assert set(designs) == set(kv.design_names()) - {"no_memory"}
    for name, d in designs.items():
        k = int(name.rsplit("_k", 1)[1])
        assert d["memory"].shape == (C * k, D)
        assert ("values" in d) == name.startswith("kv_")
        if "values" in d:
            assert d["values"].shape == d["memory"].shape
    for g in kv.GAMMAS:
        for k in kv.BUDGETS:
            b = designs[na.na_name("B", g, k)]
            for v in kv.KV_VARIANTS:
                assert torch.equal(designs[kv.kv_name(v, g, k)]["memory"], b["memory"])      # same keys as B
            assert 0.0 <= b["info"]["noisy_keys"] <= 1.0
    reference = ms.build_designs(run, torch.device("cpu"))
    for k in kv.BUDGETS:
        for base in ("kmeans", "facility_location"):
            assert torch.equal(designs[f"{base}_k{k}"]["memory"], reference[f"{base}_k{k}"]["memory"])


def test_select_candidate_prefers_smaller_gamma_and_kv_s_on_ties() -> None:
    curves = {kv.kv_name(v, g, k): [[0.5, 0.5 + (0.1 if g >= 1.0 else 0.0)]]
              for v in kv.KV_VARIANTS for g in kv.GAMMAS for k in kv.BUDGETS}
    sel = kv.select_candidate({"beta_grid": [1.0, 10.0], "beta_curves": curves})
    assert sel["candidate"] == "S"
    assert all(sel["gamma"][v][str(k)] == 1.0 for v in kv.KV_VARIANTS for k in kv.BUDGETS)


def test_end_to_end_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "load_data", lambda cfg: None)
    monkeypatch.setattr(ms, "tune_backbone", lambda *a, **k: None)
    monkeypatch.setattr(ms, "prepare_run", lambda cfg, data, seed, device, retrain=False: fake_run(seed))
    cfg = {"dataset": "synthetic", "device": "cpu", "seeds": [0, 1], "beta_values": [1.0, 10.0],
           "beta_selection_sets": [("clean", 0.0), ("gaussian", 0.05)],
           "gaussian_sigmas": [0.0, 0.1], "salt_pepper_probs": [0.0, 0.05], "occlusion_sizes": [0, 2]}
    raw = kv.run_key_value(cfg)
    assert set(raw["results"]) == set(kv.design_names()) == set(raw["train_results"])
    assert raw["decision"]["band"] in ("Pass", "Weak positive", "Fail")
    assert set(raw["design_info"]) == {n for n in kv.design_names() if n.startswith(("kv_", "na_fl_B"))}
    text = kv.build_report(raw)
    for section in ("## 1. Tune-half selection", "## 2. Decision", "mechanism", "## 4. Fraction of keys",
                    "## 5. Per-level", "## 6. Training / validation", "occlusion AUC"):
        assert section in text
