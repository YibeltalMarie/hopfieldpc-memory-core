"""Tests for Amendment 2 of the screening criteria (docs/phase1_criteria.md)."""

from __future__ import annotations

import benchmarks.screening_criteria as sc


def summary(mean_pp: float, n_pos: int, n: int = 10) -> dict:
    return {"mean": mean_pp / 100.0, "n_pos": n_pos, "n": n}


def test_original_rule_is_unchanged() -> None:
    assert sc.CLEAN_LOSS_PP == 0.3
    assert sc.band_at_budget(summary(0.6, 7), summary(-0.85, 1)) == "fail"


def test_amended_clean_margin_and_bands() -> None:
    assert sc.AMENDED_CLEAN_LOSS_PP == 1.0 and sc.PRIMARY_METRIC == "known_auc"
    assert sc.amended_band_at_budget(summary(0.6, 7), summary(-0.85, 1)) == "pass"
    assert sc.amended_band_at_budget(summary(0.6, 7), summary(-1.17, 1)) == "fail"      # clean loss > 1 pp
    assert sc.amended_band_at_budget(summary(0.6, 6), summary(0.0, 5)) == "fail"        # fewer than 7/10 seeds
    assert sc.amended_band_at_budget(summary(0.2, 8), summary(0.0, 5)) == "weak"
    assert sc.amended_band_at_budget(summary(0.05, 8), summary(0.0, 5)) == "fail"
    assert sc.method_band({"2": "pass", "4": "fail", "8": "pass", "16": "fail"}) == "Pass"


def test_known_type_auc_ignores_occlusion() -> None:
    assert abs(sc.known_type_auc({"gaussian_auc": 0.6, "salt_pepper_auc": 0.4, "occlusion_auc": 0.0}) - 0.5) < 1e-12
