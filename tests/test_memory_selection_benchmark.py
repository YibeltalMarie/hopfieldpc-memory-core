"""Tests for the statistics and verdict rule of benchmarks/memory_selection_benchmark.py."""

from __future__ import annotations

import pytest

from benchmarks.memory_selection_benchmark import (
    BUDGETS,
    NON_INFERIORITY_MARGIN,
    compare,
    design_names,
    holm_adjust,
)
from benchmarks.mnist_baseline import NOISE_TYPES


def test_holm_adjust_matches_hand_computation() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([None, 0.02]) == [None, 0.02]
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_design_names_cover_every_method_and_budget() -> None:
    names = design_names()
    assert names[0] == "class_means" and len(names) == 1 + 3 * len(BUDGETS)


def fake_raw(fl_gain: float, fl_clean: float, seeds: int = 10) -> dict:
    def entry(auc: float, clean: float) -> dict:
        return {"clean": clean, **{f"{k}_auc": auc for k in NOISE_TYPES}}

    test = {}
    for k in BUDGETS:
        # Small seed-dependent spread so deltas are distinct, all with the same sign.
        test[f"kmeans_k{k}"] = [entry(0.60 + 0.01 * s, 0.990) for s in range(seeds)]
        test[f"facility_location_k{k}"] = [
            entry(0.60 + 0.01 * s + fl_gain * (1 + 0.1 * s), 0.990 + fl_clean) for s in range(seeds)
        ]
        test[f"random_k{k}"] = [entry(0.60 + 0.01 * s, 0.990) for s in range(seeds)]
    return {"test": test}


def verdicts(rows: list[dict], method: str, metric: str) -> list[str]:
    return [r["verdict"] for r in rows if r["method"] == method and r["metric"] == metric]


def test_consistent_gain_is_a_win_and_identical_is_not() -> None:
    rows = compare(fake_raw(fl_gain=0.01, fl_clean=0.0))
    assert verdicts(rows, "facility_location", "gaussian_auc") == ["WIN"] * len(BUDGETS)
    assert verdicts(rows, "random", "gaussian_auc") == ["no significant difference"] * len(BUDGETS)


def test_clean_accuracy_drop_beyond_margin_blocks_a_win() -> None:
    rows = compare(fake_raw(fl_gain=0.01, fl_clean=-(NON_INFERIORITY_MARGIN + 0.001)))
    assert "WIN" not in verdicts(rows, "facility_location", "gaussian_auc")
    assert verdicts(rows, "facility_location", "clean") == ["inferior"] * len(BUDGETS)


def test_consistent_drop_is_a_loss() -> None:
    rows = compare(fake_raw(fl_gain=-0.01, fl_clean=0.0))
    assert verdicts(rows, "facility_location", "salt_pepper_auc") == ["LOSS"] * len(BUDGETS)


def test_too_few_seeds_never_claims_significance() -> None:
    rows = compare(fake_raw(fl_gain=0.01, fl_clean=0.0, seeds=3))
    assert all(r["wilcoxon_p"] is None and r["verdict"] != "WIN" for r in rows if r["metric"] != "clean")


def test_primary_family_is_six_tests_and_secondary_is_descriptive() -> None:
    rows = compare(fake_raw(fl_gain=0.01, fl_clean=0.0))
    fl = [r for r in rows if r["method"] == "facility_location"]
    primary = [r for r in fl if r["primary"]]
    assert len(primary) == 2 * len(BUDGETS)
    # 10 seeds, all positive: exact Wilcoxon p = 2 / 2**10; Holm over six equal p-values = 6p.
    assert all(r["holm_p"] == pytest.approx(6 * 2 / 2**10) for r in primary)
    occlusion = [r for r in fl if r["metric"] == "occlusion_auc"]
    assert all(r["holm_p"] == pytest.approx(3 * 2 / 2**10) for r in occlusion)
    assert verdicts(rows, "facility_location", "occlusion_auc") == ["higher (secondary)"] * len(BUDGETS)


def test_paired_differences_report_their_spread() -> None:
    rows = compare(fake_raw(fl_gain=0.01, fl_clean=0.0))
    row = next(r for r in rows if r["method"] == "facility_location" and r["metric"] == "gaussian_auc")
    expected = [0.01 * (1 + 0.1 * s) for s in range(10)]
    assert row["delta"] == pytest.approx(sum(expected) / 10)
    assert row["delta_sd"] > 0 and row["n_pos"] == 10 and row["n_neg"] == 0


def test_unpaired_seed_counts_are_rejected() -> None:
    raw = fake_raw(fl_gain=0.01, fl_clean=0.0)
    raw["test"]["facility_location_k2"] = raw["test"]["facility_location_k2"][:-1]
    with pytest.raises(ValueError):
        compare(raw)
