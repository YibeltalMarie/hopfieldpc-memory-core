"""Tests for the P6b diagnostic helpers (benchmarks/memory_diagnostics.py)."""

from __future__ import annotations

import pytest
import torch

from benchmarks.memory_diagnostics import (
    medoid_representatives,
    region_means,
    retrieval_stats,
    split_validation,
    subgroups,
)
from benchmarks.memory_selection import select_class_memory
from benchmarks.mnist_prototype_retrieval import HopfieldBridge, build_hopfield_net, build_memory


def labelled_blobs(n_per_class: int = 40, dim: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    means = 3.0 * torch.randn(10, dim, generator=gen)
    z = torch.cat([means[c] + torch.randn(n_per_class, dim, generator=gen) for c in range(10)])
    return z, torch.arange(10).repeat_interleave(n_per_class)


def test_split_validation_is_disjoint_complete_and_fixed() -> None:
    tune, ev = split_validation(101, seed=3)
    assert tune.numel() == 50 and ev.numel() == 51
    assert sorted(torch.cat([tune, ev]).tolist()) == list(range(101))
    again, _ = split_validation(101, seed=3)
    other, _ = split_validation(101, seed=4)
    assert torch.equal(tune, again) and not torch.equal(tune, other)


def test_medoids_are_distinct_real_rows_closest_to_centroids() -> None:
    z, y = labelled_blobs()
    mu = z.mean(dim=0)
    km, km_labels, _ = build_memory(z, y, per_class=3, center=True, seed=0)
    medoids, src = medoid_representatives(z, y, km, km_labels, mu)
    assert medoids.shape == km.shape and torch.equal(medoids, z[src])
    assert torch.equal(y[src], km_labels) and len(set(src.tolist())) == src.numel()
    # The first centroid's medoid is the most similar real point of its class.
    rows = torch.nonzero(y == 0, as_tuple=True)[0]
    cos = torch.nn.functional.cosine_similarity(z[rows] - mu, (km[0] - mu)[None], dim=1)
    assert src[0].item() == rows[int(cos.argmax())].item()


def test_medoids_never_reuse_a_point() -> None:
    z = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    y = torch.zeros(3, dtype=torch.long)
    cents = torch.tensor([[1.0, 0.0], [1.0, 0.01]])  # both closest to row 0
    _, src = medoid_representatives(z, y, cents, torch.zeros(2, dtype=torch.long), None)
    assert src.tolist() == [0, 1]


def test_region_means_average_assigned_members() -> None:
    z = torch.tensor([[1.0, 0.0], [1.0, 0.2], [0.0, 1.0], [0.2, 1.0]])
    y = torch.zeros(4, dtype=torch.long)
    picks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    means, stats = region_means(z, y, picks, torch.zeros(2, dtype=torch.long), None)
    assert torch.allclose(means, torch.tensor([[1.0, 0.1], [0.1, 1.0]]))
    assert 0.99 < stats["pick_to_mean"] <= 1.0 and stats["min_region_size"] == 2


def test_region_means_with_fl_picks_stay_class_consistent() -> None:
    z, y = labelled_blobs()
    mu = z.mean(dim=0)
    picks, labels, _ = select_class_memory(z, y, per_class=2, method="facility_location", center=mu, seed=0)
    means, _ = region_means(z, y, picks, labels, mu)
    assert means.shape == picks.shape and not torch.equal(means, picks)


def test_retrieval_stats_one_hot_and_uniform() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    targets = torch.tensor([0, 0])
    s = retrieval_stats(one_hot, labels, targets, head_pred=torch.tensor([0, 1]),
                        baseline_pred=torch.tensor([1, 1]), num_classes=2)
    assert s["n_eff"] == pytest.approx(1.0) and s["max_weight"] == pytest.approx(1.0)
    assert s["entropy"] == pytest.approx(0.0, abs=1e-9)
    # Query 0: right class retrieved, correct. Query 1: wrong class retrieved, wrong.
    assert s["acc"] == 0.5 and s["retrieval_error"] == 0.5 and s["blend_error"] == 0.0
    assert s["helped"] == 0.5 and s["hurt"] == 0.0 and s["true_class_mass"] == pytest.approx(0.5)

    uniform = torch.full((1, 4), 0.25)
    u = retrieval_stats(uniform, labels, torch.tensor([0]), head_pred=torch.tensor([1]),
                        baseline_pred=torch.tensor([0]), num_classes=2)
    assert u["n_eff"] == pytest.approx(4.0) and u["n_meaningful"] == 4
    # True-class mass 0.5 counts as majority, so a wrong head output is a blend error.
    assert u["blend_error"] == 1.0 and u["retrieval_error"] == 0.0 and u["hurt"] == 1.0


def test_subgroups_partition_by_quantile() -> None:
    margin = torch.arange(20, dtype=torch.float32)
    density = torch.arange(20, 0, -1, dtype=torch.float32)
    g = subgroups(margin, density, quantile=0.1)
    assert g["boundary"].sum() == 2 and g["rare"].sum() == 2
    assert not (g["typical"] & (g["boundary"] | g["rare"])).any()
    assert int(g["typical"].sum()) == 16


def test_selection_centre_equals_retrieval_centre() -> None:
    # P6b.1: selection and retrieval must use the same train-mean centre.
    z, y = labelled_blobs(dim=64)  # embedding size of the baseline net
    memory, labels, center = build_memory(z, y, per_class=2, center=True, seed=0, method="facility_location")
    expected, _, _ = select_class_memory(z, y, per_class=2, method="facility_location",
                                         center=z.mean(dim=0), seed=0)
    assert torch.equal(memory, expected)
    from benchmarks.mnist_baseline import build_baseline_net
    net = build_hopfield_net(build_baseline_net(0), memory, labels, beta=1.0, center=center)
    assert isinstance(net.bridge, HopfieldBridge)
    assert torch.allclose(net.bridge.center.cpu(), z.mean(dim=0))
