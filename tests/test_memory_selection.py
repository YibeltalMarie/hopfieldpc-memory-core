"""Tests for real-embedding memory selection (benchmarks/memory_selection.py)."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from benchmarks.memory_selection import (
    class_coverage,
    coverage_similarity,
    facility_location_objective,
    greedy_facility_location,
    nested_prefix,
    random_selection,
    select_class_memory,
)
from benchmarks.mnist_prototype_retrieval import build_memory


def unit_vectors(degrees: list[float]) -> torch.Tensor:
    rad = torch.tensor(degrees, dtype=torch.float64) * math.pi / 180.0
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=1)


def toy_q() -> torch.Tensor:
    # A=0, B=10, C=20 degrees (group 1); D=90, E=100 degrees (group 2).
    x = unit_vectors([0.0, 10.0, 20.0, 90.0, 100.0])
    return coverage_similarity(x, x)


def labelled_blobs(n_per_class: int = 60, dim: int = 6, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    means = 3.0 * torch.randn(10, dim, generator=gen)
    z = torch.cat([means[c] + torch.randn(n_per_class, dim, generator=gen) for c in range(10)])
    labels = torch.arange(10).repeat_interleave(n_per_class)
    return z.float(), labels


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def test_coverage_similarity_is_shifted_cosine_in_unit_interval() -> None:
    gen = torch.Generator().manual_seed(0)
    ref, cand, mu = torch.randn(7, 5, generator=gen), torch.randn(4, 5, generator=gen), torch.randn(5, generator=gen)
    q = coverage_similarity(ref, cand, mu)
    cos = torch.nn.functional.cosine_similarity(
        (ref - mu).double()[:, None, :], (cand - mu).double()[None, :, :], dim=-1
    )
    assert q.shape == (7, 4) and q.dtype == torch.float64
    assert torch.allclose(q, (1 + cos) / 2, atol=1e-12)
    assert torch.all((q >= 0) & (q <= 1))
    assert torch.allclose(torch.diagonal(coverage_similarity(ref, ref, mu)), torch.ones(7, dtype=torch.float64))


def test_zero_norm_centred_vector_gets_cosine_zero_like_retrieval() -> None:
    mu = torch.tensor([1.0, 2.0])
    q = coverage_similarity(torch.tensor([[1.0, 2.0], [3.0, 2.0]]), torch.tensor([[1.0, 2.0]]), mu)
    assert torch.allclose(q, torch.full((2, 1), 0.5, dtype=torch.float64))


def test_similarity_rejects_bad_shapes_and_nan() -> None:
    with pytest.raises(ValueError):
        coverage_similarity(torch.zeros(3, 2), torch.zeros(3, 4))
    with pytest.raises(ValueError):
        coverage_similarity(torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(3))
    with pytest.raises(ValueError):
        coverage_similarity(torch.tensor([[float("nan"), 1.0]]), torch.ones(1, 2))


# ---------------------------------------------------------------------------
# Greedy facility location
# ---------------------------------------------------------------------------


def test_worked_example_picks_centre_then_other_group() -> None:
    q = toy_q()
    idx, gains = greedy_facility_location(q, 3)
    # C covers everyone best; then one of the uncovered group (D and E tie on
    # paper, so rounding may pick either); the last pick only tops up group 1.
    # Exact tie-breaking is tested on exact duplicates below.
    assert idx[0].item() == 2 and idx[1].item() in (3, 4) and idx[2].item() in (0, 1)
    assert torch.allclose(gains, torch.tensor([4.2201, 0.7346, 0.0302], dtype=torch.float64), atol=1e-4)


def test_gains_equal_objective_differences_and_diminish() -> None:
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(40, 4, generator=gen)
    q = coverage_similarity(x, x)
    idx, gains = greedy_facility_location(q, 6)
    values = [facility_location_objective(q, idx[:t].tolist()) for t in range(7)]
    diffs = torch.tensor([values[t + 1] - values[t] for t in range(6)], dtype=torch.float64)
    assert torch.allclose(gains, diffs, atol=1e-9)
    assert torch.all(gains[1:] <= gains[:-1] + 1e-12)  # diminishing returns along the greedy path
    assert len(set(idx.tolist())) == 6


def test_greedy_meets_one_minus_one_over_e_bound() -> None:
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(12, 3, generator=gen)
    q = coverage_similarity(x, x)
    greedy_value = facility_location_objective(q, greedy_facility_location(q, 3)[0].tolist())
    optimum = max(facility_location_objective(q, list(s)) for s in itertools.combinations(range(12), 3))
    assert greedy_value >= (1 - 1 / math.e) * optimum
    assert greedy_value <= optimum + 1e-12


def test_selection_is_invariant_to_positive_scaling_of_q() -> None:
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(30, 5, generator=gen)
    q = coverage_similarity(x, x)
    assert torch.equal(greedy_facility_location(q, 5)[0], greedy_facility_location(0.5 * q, 5)[0])


def test_single_pick_is_most_aligned_with_mean_direction() -> None:
    # sum_i cos(z_i, m) = (sum_i unit(z_i)) . unit(m), so k=1 picks the point
    # whose direction is closest to the mean unit vector.
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(50, 4, generator=gen).double() + torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    units = torch.nn.functional.normalize(x, dim=1)
    expected = int(torch.argmax(units @ units.mean(dim=0)).item())
    assert greedy_facility_location(coverage_similarity(x, x), 1)[0].item() == expected


def test_one_pick_per_separated_cluster() -> None:
    gen = torch.Generator().manual_seed(5)
    directions = unit_vectors([0.0, 120.0, 240.0])
    x = torch.cat([d + 0.05 * torch.randn(20, 2, generator=gen, dtype=torch.float64) for d in directions])
    idx, _ = greedy_facility_location(coverage_similarity(x, x), 3)
    assert sorted(i // 20 for i in idx.tolist()) == [0, 1, 2]


def test_eligible_mask_restricts_picks_but_not_coverage() -> None:
    q = toy_q()
    eligible = torch.tensor([True, True, False, True, True])  # forbid C
    idx, gains = greedy_facility_location(q, 2, eligible=eligible)
    assert 2 not in idx.tolist()
    assert gains[0] == pytest.approx(q[:, idx[0]].sum().item())  # all five points still counted
    with pytest.raises(ValueError):
        greedy_facility_location(q, 5, eligible=eligible)


def test_exact_duplicates_break_ties_to_lowest_index() -> None:
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    idx, _ = greedy_facility_location(coverage_similarity(x, x), 2)
    assert idx.tolist() == [0, 2]


def test_greedy_rejects_invalid_input() -> None:
    q = toy_q()
    with pytest.raises(ValueError):
        greedy_facility_location(q, 0)
    with pytest.raises(ValueError):
        greedy_facility_location(q, 6)
    with pytest.raises(ValueError):
        greedy_facility_location(q - 1.0, 1)
    with pytest.raises(ValueError):
        greedy_facility_location(q, 1, eligible=torch.ones(5))


def test_random_selection_is_distinct_and_seeded() -> None:
    a = random_selection(100, 8, generator=torch.Generator().manual_seed(0))
    b = random_selection(100, 8, generator=torch.Generator().manual_seed(0))
    c = random_selection(100, 8, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b) and not torch.equal(a, c) and len(set(a.tolist())) == 8
    with pytest.raises(ValueError):
        random_selection(5, 6, generator=torch.Generator().manual_seed(0))


# ---------------------------------------------------------------------------
# Class-wise memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["facility_location", "random"])
def test_class_memory_stores_real_labelled_rows(method: str) -> None:
    z, labels = labelled_blobs()
    memory, mem_labels, source = select_class_memory(
        z, labels, per_class=4, method=method, center=z.mean(dim=0), seed=0
    )
    assert memory.shape == (40, z.shape[1]) and memory.dtype == z.dtype
    assert mem_labels.tolist() == sorted(list(range(10)) * 4)
    assert torch.equal(memory, z[source])           # real embeddings, traceable to their rows
    assert torch.equal(labels[source], mem_labels)  # each pattern keeps its own label
    assert len(set(source.tolist())) == 40


def test_class_memory_rejects_unknown_method_and_small_class() -> None:
    z, labels = labelled_blobs(n_per_class=3)
    with pytest.raises(ValueError):
        select_class_memory(z, labels, per_class=2, method="kmeans", center=None, seed=0)
    with pytest.raises(ValueError):
        select_class_memory(z, labels, per_class=4, method="random", center=None, seed=0)


def test_build_memory_methods() -> None:
    z, labels = labelled_blobs()
    fl, fl_labels, center = build_memory(z, labels, per_class=3, center=True, seed=0,
                                         method="facility_location")
    again, _, _ = build_memory(z, labels, per_class=3, center=True, seed=7, method="facility_location")
    assert center is not None and torch.allclose(center, z.mean(dim=0))
    assert torch.equal(fl, again)  # facility location is deterministic, independent of seed
    assert fl_labels.tolist() == sorted(list(range(10)) * 3)
    assert all(bool((z == row).all(dim=1).any()) for row in fl)

    r0, _, _ = build_memory(z, labels, per_class=3, center=True, seed=0, method="random")
    r1, _, _ = build_memory(z, labels, per_class=3, center=True, seed=1, method="random")
    assert not torch.equal(r0, r1)

    km_default, _, _ = build_memory(z, labels, per_class=3, center=True, seed=0)
    km_explicit, _, _ = build_memory(z, labels, per_class=3, center=True, seed=0, method="kmeans")
    assert torch.equal(km_default, km_explicit)  # existing behaviour unchanged
    with pytest.raises(ValueError):
        build_memory(z, labels, per_class=3, center=True, seed=0, method="medoids")


@pytest.mark.parametrize("method", ["facility_location", "random"])
def test_selection_is_nested_across_budgets(method: str) -> None:
    z, labels = labelled_blobs()
    mu = z.mean(dim=0)
    big, big_labels, _ = select_class_memory(z, labels, per_class=6, method=method, center=mu, seed=3)
    small, small_labels, _ = select_class_memory(z, labels, per_class=2, method=method, center=mu, seed=3)
    prefix, prefix_labels = nested_prefix(big, big_labels, per_class_max=6, per_class=2)
    assert torch.equal(prefix, small) and torch.equal(prefix_labels, small_labels)
    with pytest.raises(ValueError):
        nested_prefix(big, big_labels, per_class_max=6, per_class=7)


def test_class_coverage_uses_only_same_class_patterns() -> None:
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    memory_labels = torch.tensor([0, 1])
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1])
    cov = class_coverage(z, labels, memory, memory_labels, None)
    # Point 1 is identical to the class-0 pattern but must be scored against class 1.
    assert torch.allclose(cov, torch.tensor([1.0, 0.5], dtype=torch.float64))
    with pytest.raises(ValueError):
        class_coverage(z, torch.tensor([0, 2]), memory, memory_labels, None)
