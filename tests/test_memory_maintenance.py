"""Tests for the RQ2 maintenance policies and benchmark helpers."""

from __future__ import annotations

import itertools

import pytest
import torch

from benchmarks.memory_maintenance import POLICY_CLASSES, FLSwap, RunningAverageLayout, stream_split, unit
from benchmarks.memory_maintenance_benchmark import (
    CHECKPOINTS,
    START_SIZES,
    STREAM_PER_CLASS,
    comparisons,
    run_stream,
    stream_plan,
)

C, K, D = 3, 4, 6


def toy(seed: int = 0, n_init: int = 12):
    g = torch.Generator().manual_seed(seed)
    centers = 3.0 * torch.randn(C, D, generator=g)
    init_y = torch.arange(C).repeat_interleave(n_init)
    init_z = centers[init_y] + torch.randn(init_y.numel(), D, generator=g)
    patterns = torch.stack([init_z[init_y == c][:K] for c in range(C)])
    counts = torch.ones(C, K)
    mu = init_z.mean(dim=0)
    stream_y = torch.randint(0, C, (120,), generator=g)
    stream_z = centers[stream_y] + torch.randn(120, D, generator=g)
    return patterns, counts, mu, init_z, init_y, stream_z, stream_y


def make(name: str, seed: int = 0, **kw):
    patterns, counts, mu, init_z, init_y, sz, sy = toy(seed)
    kw.setdefault("ref_size", 20)
    kw.setdefault("layout_every", 10)
    return POLICY_CLASSES[name](patterns, counts, mu, init_z, init_y, seed=seed, **kw), sz, sy


# -- stream_split ------------------------------------------------------------

def test_stream_split_is_disjoint_balanced_and_deterministic() -> None:
    labels = torch.arange(4).repeat_interleave(30)[torch.randperm(120, generator=torch.Generator().manual_seed(1))]
    init, stream = stream_split(labels, 5, 20, seed=9)
    assert torch.equal(torch.bincount(labels[init], minlength=4), torch.full((4,), 5))
    assert torch.equal(torch.bincount(labels[stream], minlength=4), torch.full((4,), 20))
    assert not set(init.tolist()) & set(stream.tolist())
    i2, s2 = stream_split(labels, 5, 20, seed=9)
    assert torch.equal(init, i2) and torch.equal(stream, s2)
    assert not torch.equal(stream, stream_split(labels, 5, 20, seed=10)[1])


def test_stream_split_follows_one_order_and_nests_starts() -> None:
    labels = torch.arange(3).repeat_interleave(40)
    order = torch.randperm(120, generator=torch.Generator().manual_seed(4))
    pos = torch.empty_like(order)
    pos[order] = torch.arange(120)
    init, stream = stream_split(labels, 6, 20, seed=4)
    assert (pos[stream].diff() > 0).all()          # interleaved in the seeded order
    for c in range(3):                             # initial rows precede the class's stream rows
        assert pos[init[labels[init] == c]].max() < pos[stream[labels[stream] == c]].min()
    small, _ = stream_split(labels, 3, 20, seed=4)
    assert set(small.tolist()) < set(init.tolist())


def test_stream_split_rejects_small_classes() -> None:
    with pytest.raises(ValueError):
        stream_split(torch.arange(2).repeat_interleave(10), 5, 6, seed=0)


def test_stream_plan_full_and_reduced() -> None:
    assert stream_plan(5000) == (STREAM_PER_CLASS, list(START_SIZES))
    per_class, starts = stream_plan(200)
    assert per_class == 100 and starts == [10, 30, 100]


# -- policies ------------------------------------------------------------------

@pytest.mark.parametrize("name", list(POLICY_CLASSES))
def test_capacity_and_invariants_hold_for_every_policy(name: str) -> None:
    policy, sz, sy = make(name)
    for z, y in zip(sz, sy):
        policy.observe(z, int(y))
        policy.check_invariants()
    memory, labels = policy.memory()
    assert memory.shape == (C * K, D) and memory.dtype == torch.float32
    assert torch.equal(labels, torch.arange(C).repeat_interleave(K))
    assert policy.stats["arrivals"] == sy.numel()
    assert all(r.shape[0] <= 20 for r in policy.ref_raw)


def test_frozen_never_changes() -> None:
    policy, sz, sy = make("frozen")
    before = policy.memory()[0]
    for z, y in zip(sz, sy):
        assert policy.observe(z, int(y))["action"] == "keep"
    assert torch.equal(before, policy.memory()[0]) and policy.stats["changed"] == 0


def test_fifo_cycles_slots_of_the_arrival_class() -> None:
    policy, sz, _ = make("fifo")
    slots = [policy.observe(sz[t], 1)["slot"] for t in range(2 * K + 1)]
    assert slots == [j % K for j in range(2 * K + 1)]
    assert torch.allclose(policy.P[1, 0], sz[2 * K].double())
    assert torch.equal(policy.P[0], make("fifo")[0].P[0])  # other classes untouched


def test_reservoir_is_seeded_and_stores_real_embeddings() -> None:
    a, sz, sy = make("reservoir", seed=3)
    b, _, _ = make("reservoir", seed=3)
    for z, y in zip(sz, sy):
        assert a.observe(z, int(y)) == b.observe(z, int(y))
    initial = make("reservoir", seed=3)[0].P
    real = torch.cat([sz.double(), initial.reshape(-1, D)])
    for row in a.P.reshape(-1, D):
        assert (real - row).abs().sum(dim=1).min() < 1e-12


def test_fl_swap_gains_match_brute_force() -> None:
    policy, sz, sy = make("fl_swap")
    for t in range(15):
        policy.observe(sz[t], int(sy[t]))
    c = 2
    u = unit(sz[40], policy.mu)
    gains = policy.swap_gains(c, u)
    base = policy.coverage(c)
    for j in range(K):
        U_new = policy.U[c].clone()
        U_new[j] = u
        assert gains[j].item() == pytest.approx(policy.coverage(c, U_new) - base, abs=1e-9)


def test_fl_swap_keeps_memory_when_no_gain() -> None:
    policy, _, _ = make("fl_swap", epsilon=1e-9)  # tolerance for rounding in the zero gain
    c, before = 0, policy.P.clone()
    decision = policy.observe(policy.P[c, 0].float(), c)  # duplicate of a stored pattern adds nothing
    assert decision["action"] == "keep" and abs(decision["delta"]) <= 1e-9
    assert torch.equal(before, policy.P)


def test_fl_swap_coverage_never_decreases_on_a_fixed_reference() -> None:
    policy, sz, sy = make("fl_swap", ref_size=10_000)
    for z, y in zip(sz, sy):
        c = int(y)
        before = policy.coverage(c)
        d = policy._update(c, z.double(), unit(z, policy.mu))  # reference fixed during the step
        assert policy.coverage(c) >= before - 1e-9
        assert (d["action"] == "replace") == (d["delta"] > 0)


def test_running_average_equals_exact_mean_of_absorbed_points() -> None:
    policy, sz, sy = make("ra")
    init = policy.P.clone()
    members = {(c, j): [init[c, j]] for c in range(C) for j in range(K)}
    for z, y in zip(sz, sy):
        d = policy.observe(z, int(y))
        assert d["action"] == "absorb"
        members[(int(y), d["slot"])].append(z.double())
    for (c, j), pts in members.items():
        assert torch.allclose(policy.P[c, j], torch.stack(pts).mean(dim=0), atol=1e-10)
        assert policy.n[c, j] == len(pts)
    assert policy.stats["absorbed"] == sy.numel() and policy.stats["layout_checks"] == 0


def test_running_average_absorbs_into_nearest_same_class_slot() -> None:
    policy, sz, _ = make("ra")
    u = unit(sz[0], policy.mu)
    expected = int(torch.argmax(policy.U[1] @ u))
    assert policy.observe(sz[0], 1)["slot"] == expected


def test_layout_step_commits_only_when_coverage_rises() -> None:
    policy, sz, sy = make("ra_fl", layout_every=10**9)  # drive layout manually
    for z, y in zip(sz, sy):
        policy.observe(z, int(y))
    for c in range(C):
        before, P_before, n_before = policy.coverage(c), policy.P[c].clone(), policy.n[c].sum()
        change = policy.layout_step(c)
        after = policy.coverage(c)
        if change is None:
            assert torch.equal(P_before, policy.P[c]) and after == before
        else:
            assert after > before
            assert change["layout_delta"] == pytest.approx(after - before, abs=1e-9)
            a, b = change["merged"]
            assert policy.n[c, b] == 1 and policy.n[c].sum() == n_before + 1  # merged counts kept, spawn adds 1
            assert any(torch.allclose(policy.P[c, b], r) for r in policy.ref_raw[c])


def test_layout_step_merges_near_duplicates_and_spawns_uncovered_point() -> None:
    mu = torch.zeros(2)
    e = [torch.tensor([1.0, 0.0]), torch.tensor([0.999, 0.045]), torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])]
    patterns = torch.stack([e[0], e[1], e[2]])[None]                # slots 0 and 1 nearly identical
    init_z = torch.stack([e[0], e[2], e[3], e[3], e[3]])           # many reference points at e3, uncovered
    policy = RunningAverageLayout(patterns, torch.ones(1, 3), mu, init_z, torch.zeros(5, dtype=torch.long),
                                  seed=0, ref_size=50, layout_every=10**9)
    change = policy.layout_step(0)
    assert change is not None and sorted(change["merged"]) == [0, 1]
    assert torch.allclose(policy.P[0, change["spawn_slot"]], e[3].double())


def test_layout_runs_every_layout_every_class_arrivals() -> None:
    policy, sz, _ = make("ra_fl", layout_every=5)
    for t in range(23):
        policy.observe(sz[t], 0)
    assert policy.stats["layout_checks"] == 4


def test_decision_log_and_conflicts() -> None:
    policy, sz, sy = make("fl_swap", log=True)
    for z, y in zip(sz, sy):
        policy.observe(z, int(y))
    assert len(policy.log) == sy.numel()
    assert [e["t"] for e in policy.log] == list(range(1, sy.numel() + 1))
    assert {"action", "slot", "delta", "class", "conflict"} <= set(policy.log[0])
    assert policy.stats["conflicts"] == sum(e["conflict"] for e in policy.log)
    assert make("ra")[0].log is None


@pytest.mark.parametrize("name", list(POLICY_CLASSES))
def test_zero_norm_arrival_is_handled(name: str) -> None:
    policy, _, _ = make(name)
    policy.observe(policy.mu.float().clone(), 1)  # centred vector is zero
    policy.check_invariants()


def test_policies_reject_bad_shapes() -> None:
    patterns, counts, mu, init_z, init_y, _, _ = toy()
    with pytest.raises(ValueError):
        FLSwap(patterns.reshape(C * K, D), counts, mu, init_z, init_y, seed=0)


# -- benchmark helpers -----------------------------------------------------------

def test_run_stream_snapshots_match_checkpoints() -> None:
    policy, sz, sy = make("fifo")
    snaps, us = run_stream(policy, sz, sy)
    assert len(snaps) == len(CHECKPOINTS) and us >= 0
    assert torch.equal(snaps[-1], policy.memory()[0])
    replay, _, _ = make("fifo")
    stop = round(CHECKPOINTS[0] * sy.numel())
    for t in range(stop):
        replay.observe(sz[t], int(sy[t]))
    assert torch.equal(snaps[0], replay.memory()[0])


def test_comparisons_verdicts_and_holm_families() -> None:
    seeds = list(range(10))
    designs = ("frozen", "fifo", "reservoir", "fl_swap", "ra", "ra_fl")
    shift = {"frozen": 0.0, "fifo": 0.0, "reservoir": 0.0, "fl_swap": 0.0, "ra": 0.02, "ra_fl": 0.02}

    def rows(name, s):
        noise = 0.001 * ((s * 7 + designs.index(name)) % 3)
        return [{"clean": 0.98, "gaussian_auc": 0.8 + shift[name] + noise, "salt_pepper_auc": 0.7 + shift[name] + noise,
                 "occlusion_auc": 0.6}] * 3

    raw = {"seeds": seeds, "starts": [10, 30],
           "test": {n: {n0: {s: rows(n, s) for s in seeds} for n0 in (10, 30)} for n in designs}}
    out = comparisons(raw)
    assert len(out) == 5 * 4
    primary = [r for r in out if r["primary"]]
    assert len(primary) == 4 and all(r["verdict"] == "WIN" for r in primary)
    assert all(r["verdict"] != "WIN" for r in out if r["vs"] == "ra")
    for vs, group in itertools.groupby(out, key=lambda r: r["vs"]):
        group = list(group)
        assert len(group) == 4
        assert all(r["holm_p"] is None or r["holm_p"] >= r["wilcoxon_p"] for r in group)
