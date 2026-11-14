"""Tests for the noise-level calibration rule (pure function; no data or training)."""

from __future__ import annotations

import pytest

from benchmarks.cifar10_baseline import DEFAULT_CONFIG_PATH
from benchmarks.noise_calibration import DENSE_LEVELS, calibrate_grid, main, yaml_snippet

LEVELS = [0.01 * i for i in range(21)]
ACCS = [0.9 - 0.06 * i if i < 12 else 0.18 - 0.002 * (i - 12) for i in range(21)]  # crosses 0.20 at index 12


def test_grid_ends_at_first_level_at_or_below_the_floor() -> None:
    r = calibrate_grid(LEVELS, ACCS)
    assert r["floor_reached"] is True
    assert r["grid"][0] == 0 and r["grid"][-1] == LEVELS[12]
    assert len(r["grid"]) == 6
    assert all(b > a for a, b in zip(r["grid"], r["grid"][1:]))
    assert r["end_accuracy"] == ACCS[12]


def test_interior_levels_follow_equal_accuracy_steps() -> None:
    r = calibrate_grid(LEVELS, ACCS)
    steps = [a - b for a, b in zip(r["grid_accuracy"], r["grid_accuracy"][1:])]
    ideal = (ACCS[0] - ACCS[12]) / 5
    assert all(abs(s - ideal) <= 0.06 + 1e-9 for s in steps)   # within one dense step of the ideal


def test_selection_level_is_off_grid_and_nearest_the_midpoint() -> None:
    r = calibrate_grid(LEVELS, ACCS)
    assert r["selection_level"] not in r["grid"]
    mid = (ACCS[0] + ACCS[12]) / 2
    candidates = [i for i in range(1, 12) if LEVELS[i] not in r["grid"]]
    best = min(abs(ACCS[i] - mid) for i in candidates)
    assert abs(r["selection_accuracy"] - mid) == pytest.approx(best)


def test_floor_not_reached_uses_the_largest_level() -> None:
    accs = [0.9 - 0.02 * i for i in range(13)]
    r = calibrate_grid([2.0 * i for i in range(13)], accs)
    assert r["floor_reached"] is False and r["grid"][-1] == 24.0


def test_short_curves_give_shorter_grids_without_duplicates() -> None:
    r = calibrate_grid([0.0, 0.1, 0.2], [0.9, 0.5, 0.1])
    assert r["grid"] == [0.0, 0.1, 0.2] and r["selection_level"] is None


@pytest.mark.parametrize("levels, accs", [
    ([0.0, 0.1], [0.9, 0.1]),                  # too short
    ([0.1, 0.2, 0.3], [0.9, 0.5, 0.1]),        # does not start at 0
    ([0.0, 0.2, 0.1], [0.9, 0.5, 0.1]),        # not increasing
    ([0.0, 0.1, 0.2], [0.9, 0.5]),             # length mismatch
    ([0.0, 0.1, 0.2], [0.1, 0.05, 0.0]),       # already at the floor
])
def test_invalid_inputs_are_rejected(levels, accs) -> None:
    with pytest.raises(ValueError):
        calibrate_grid(levels, accs)


def test_dense_levels_start_clean_and_increase() -> None:
    for levels in DENSE_LEVELS.values():
        assert levels[0] == 0 and all(b > a for a, b in zip(levels, levels[1:]))


def test_yaml_snippet_formats_the_config_lines() -> None:
    result = {
        "gaussian": {"grid": [0.0, 0.03, 0.13], "selection_level": 0.06},
        "salt_pepper": {"grid": [0.0, 0.01, 0.12], "selection_level": 0.02},
        "occlusion": {"grid": [0.0, 10.0, 24.0], "selection_level": 16.0},
    }
    text = yaml_snippet(result)
    assert 'beta_selection_sets: [["clean", 0], ["gaussian", 0.06], ["salt_pepper", 0.02]]' in text
    assert "occlusion_sizes: [0, 10, 24]" in text and "gaussian_sigmas: [0, 0.03, 0.13]" in text


def test_main_refuses_to_run_without_cached_backbones(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["--config", str(DEFAULT_CONFIG_PATH), "--reports-dir", str(tmp_path)])
