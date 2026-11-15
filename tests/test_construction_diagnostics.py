"""Tests for the Stage 1 construction diagnostics (synthetic data; no dataset or training)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import benchmarks.construction_diagnostics as cd

C, PER, D = 10, 30, 12   # build_memory assumes 10 classes


def clusters(seed: int = 0, spread: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """C tight, well-separated clusters (one per class) around orthogonal directions."""
    g = torch.Generator().manual_seed(seed)
    centers = 5.0 * torch.eye(C, D)
    y = torch.arange(C).repeat_interleave(PER)
    return centers[y] + spread * torch.randn(C * PER, D, generator=g), y


def test_centred_unit_rows_have_unit_norm() -> None:
    z, _ = clusters()
    u = cd.centred_unit(z, z.mean(dim=0))
    assert torch.allclose(u.norm(dim=1), torch.ones(u.shape[0]), atol=1e-6)


def test_knn_matches_brute_force_with_blocks_and_self_exclusion() -> None:
    z, _ = clusters(spread=1.0)
    u = cd.centred_unit(z, z.mean(dim=0))
    sims, idx = cd.knn(u, u, 5, exclude_self=True, block=7)
    full = u @ u.T
    full.fill_diagonal_(float("-inf"))
    ref_sims, _ = full.topk(5, dim=1)
    assert torch.allclose(sims, ref_sims, atol=1e-6)
    assert not (idx == torch.arange(u.shape[0])[:, None]).any()


def test_knn_rejects_bad_k() -> None:
    u = cd.centred_unit(torch.randn(5, D), torch.zeros(D))
    with pytest.raises(ValueError):
        cd.knn(u, u, 5, exclude_self=True)
    with pytest.raises(ValueError):
        cd.knn(u[:3], u, 2, exclude_self=True)


def test_boundary_zero_for_separated_classes_and_high_for_intruder() -> None:
    z, y = clusters()
    center = z.mean(dim=0)
    u = cd.centred_unit(z, center)
    b = cd.boundary_scores(u, y, u, y, kappa=10, same_set=True)
    assert torch.all(b == 0)
    y2 = y.clone()
    y2[0] = 1                       # a class-1 label placed inside the class-0 cluster
    b2 = cd.boundary_scores(u, y2, u, y2, kappa=10, same_set=True)
    assert b2[0].item() == pytest.approx(1.0)


def test_density_is_higher_in_a_tighter_class() -> None:
    g = torch.Generator().manual_seed(1)
    y = torch.tensor([0] * 40 + [1] * 40)
    base = torch.zeros(80, D)
    base[:40, 0], base[40:, 1] = 5.0, 5.0
    noise = torch.randn(80, D, generator=g)
    noise[:40] *= 0.05              # class 0 tight
    noise[40:] *= 1.5               # class 1 spread out
    z = base + noise
    rho = cd.density_proxy(cd.centred_unit(z, z.mean(dim=0)), y, kappa=10)
    assert rho[:40].mean() > rho[40:].mean()
    assert torch.all((rho >= 0) & (rho <= 1))


def test_rankdata_averages_ties() -> None:
    r = cd.rankdata(torch.tensor([3.0, 1.0, 2.0, 2.0]))
    assert r.tolist() == [4.0, 1.0, 2.5, 2.5]


def test_spearman_extremes_and_constant() -> None:
    a = torch.arange(10.0)
    assert cd.spearman(a, a ** 3) == pytest.approx(1.0)
    assert cd.spearman(a, -a) == pytest.approx(-1.0)
    assert math.isnan(cd.spearman(a, torch.ones(10)))


def test_neighbour_stats_and_nearest_pattern_accuracy() -> None:
    z, y = clusters()
    center = z.mean(dim=0)
    u = cd.centred_unit(z, center)
    stats = cd.neighbour_stats(u, y, u, y, kappa=5)
    assert stats == {"hit": 1.0, "top1": 1.0, "purity": 1.0}
    means = torch.stack([z[y == c].mean(dim=0) for c in range(C)])
    labels = torch.arange(C)
    assert cd.nearest_pattern_accuracy(u, y, cd.centred_unit(means, center), labels) == 1.0
    assert cd.nearest_pattern_accuracy(u, y, cd.centred_unit(means, center), labels.roll(1)) == 0.0


def test_corrupt_per_image_types_levels_and_determinism() -> None:
    images = torch.full((4, 3, 5, 5), 0.5)
    kinds = torch.tensor([0, 0, 1, 1])
    levels = torch.tensor([0.0, 0.2, 0.0, 0.5])
    out = cd.corrupt_per_image(images, kinds, levels, torch.Generator().manual_seed(3))
    again = cd.corrupt_per_image(images, kinds, levels, torch.Generator().manual_seed(3))
    assert torch.equal(out, again)
    assert torch.equal(out[0], images[0]) and torch.equal(out[2], images[2])   # level 0: unchanged
    assert not torch.equal(out[1], images[1]) and out.min() >= 0 and out.max() <= 1
    changed = out[3] != 0.5
    assert changed.any() and torch.all((out[3][changed] == 0) | (out[3][changed] == 1))
    assert torch.all(changed.all(dim=0) == changed.any(dim=0))                  # same pixels on every channel


def fake_run(seed: int = 0, n_val: int = 40) -> SimpleNamespace:
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(C).repeat_interleave(PER)
    pattern = torch.eye(C, 16)[y].view(-1, 1, 4, 4)
    images = (0.5 + 0.3 * pattern + 0.01 * torch.randn(C * PER, 1, 4, 4, generator=g)).clamp(0, 1)
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(16, D))
    net = SimpleNamespace(encoder=encoder)
    with torch.no_grad():
        z = encoder(images)
    yv = torch.arange(n_val) % C
    zv = z[yv * PER]
    noise = torch.randn(n_val, D, generator=g)
    run = SimpleNamespace(seed=seed, z_train=z, z_val=zv, net=net, batch_size=16, device=torch.device("cpu"),
                          splits=SimpleNamespace(train_labels=y, val_labels=yv, train_images=images))
    run.val_embeddings = lambda kind, level: zv if level == 0 else zv + 0.5 * float(level) * noise
    return run


GRIDS = {"gaussian": [0.0, 0.1, 0.2], "salt_pepper": [0.0, 0.05], "occlusion": [0, 2]}


def test_diagnose_seed_structure_and_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cd, "BUDGETS", (1, 2))
    monkeypatch.setattr(cd, "DECISION_K", 2)
    row = cd.diagnose_seed(fake_run(), GRIDS, torch.device("cpu"), kappa=5)
    kinds = [lv["kind"] for lv in row["d1"]["levels"]]
    assert kinds == ["gaussian", "gaussian", "salt_pepper", "occlusion"]          # level 0 skipped
    assert set(row["d2"]) == {"1", "2"} and all(0 <= v <= 1 for v in row["d2"].values())
    assert 0 <= row["d3"]["B_train"]["mean"] <= 1 and len(row["d3"]["per_class"]) == C
    raw = {"dataset": "synthetic", "seeds": [0, 1], "kappa": 5,
           "per_seed": [row, cd.diagnose_seed(fake_run(1), GRIDS, torch.device("cpu"), kappa=5)]}
    dec = cd.decisions(raw)
    assert dec["d1"]["triggered"] == (dec["d1"]["corrupted_hit"][0] >= dec["d1"]["clean_hit"][0])
    assert dec["d2"]["triggered"] == (dec["d2"]["accuracy_k8"][0] > cd.CA_ACC_THRESHOLD)
    report = cd.build_report(raw)
    assert "## Decisions" in report and "D1 noise-aware premise" in report and "salt_pepper 0.05" in report


def test_decision_hit_ignores_occlusion() -> None:
    row = {"d1": {"levels": [{"kind": "gaussian", "hit": 0.8}, {"kind": "salt_pepper", "hit": 0.6},
                             {"kind": "occlusion", "hit": 0.0}]}}
    assert cd.decision_hit(row) == pytest.approx(0.7)
