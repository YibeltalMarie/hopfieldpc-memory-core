"""Decision criteria for screening runs: Pass / Weak positive / Fail vs k-means (docs/phase1_criteria.md).

A method is compared with class-wise k-means at the same budget k, paired by
seed, on noise AUC (primary) and clean accuracy (guard). Inputs are
``memory_diagnostics.paired_summary`` dicts of differences in accuracy units
(method - k-means); thresholds are in percentage points (pp).

Per budget:  fail  if fewer than 70% of seeds are positive on noise AUC, or
                   the clean-accuracy loss exceeds 0.3 pp;
             pass  if the mean noise-AUC gain is >= +0.3 pp;
             weak  if it is in [+0.1, +0.3) pp;
             fail  otherwise.
Per method:  Pass if >= 2 budgets pass; Weak positive if >= 2 budgets are
             pass or weak; Fail otherwise.
"""

from __future__ import annotations

import math
from typing import Any

PASS_PP = 0.3
WEAK_PP = 0.1
MIN_POS_FRAC = 0.7
MIN_BUDGETS = 2
CLEAN_LOSS_PP = 0.3


def band_at_budget(noise: dict[str, Any], clean: dict[str, Any]) -> str:
    """'pass', 'weak' or 'fail' at one budget from paired noise-AUC and clean-accuracy differences."""
    gain_pp = 100.0 * noise["mean"]
    clean_pp = 100.0 * clean["mean"]
    if noise["n_pos"] < math.ceil(MIN_POS_FRAC * noise["n"] - 1e-9) or clean_pp < -CLEAN_LOSS_PP:
        return "fail"
    if gain_pp >= PASS_PP:
        return "pass"
    if gain_pp >= WEAK_PP:
        return "weak"
    return "fail"


def method_band(per_budget: dict[Any, str]) -> str:
    """'Pass', 'Weak positive' or 'Fail' for a method from its per-budget bands."""
    bands = list(per_budget.values())
    if bands.count("pass") >= MIN_BUDGETS:
        return "Pass"
    if sum(b in ("pass", "weak") for b in bands) >= MIN_BUDGETS:
        return "Weak positive"
    return "Fail"


# --- Amendment 2 (2026-11-24; docs/phase1_criteria.md) -------------------------------------------
# For methods screened after this amendment. The functions above stay unchanged, so the
# recorded Stage 2-4 decisions remain reproducible. Primary metric: known-type AUC (mean of the
# Gaussian and salt-and-pepper AUCs, as in the research plan P4.3); occlusion and the three-type
# noise AUC are secondary. Clean guard: mean clean-accuracy loss <= AMENDED_CLEAN_LOSS_PP
# (provisional, pending the project leader's confirmation).
AMENDED_CLEAN_LOSS_PP = 1.0
PRIMARY_METRIC = "known_auc"


def known_type_auc(result: dict[str, Any]) -> float:
    """Primary metric under Amendment 2: mean of the Gaussian and salt-and-pepper AUCs of one result."""
    return (result["gaussian_auc"] + result["salt_pepper_auc"]) / 2


def amended_band_at_budget(primary: dict[str, Any], clean: dict[str, Any]) -> str:
    """Amendment 2 band at one budget from paired known-type-AUC and clean-accuracy differences.

    Same bands and seed rule as before; only the primary metric and the clean margin change.
    """
    if primary["n_pos"] < MIN_POS_FRAC * primary["n"] - 1e-9 or 100.0 * clean["mean"] < -AMENDED_CLEAN_LOSS_PP:
        return "fail"
    gain_pp = 100.0 * primary["mean"]
    if gain_pp >= PASS_PP:
        return "pass"
    if gain_pp >= WEAK_PP:
        return "weak"
    return "fail"
