# Phase 1 decision criteria (memory construction)

Fixed on 2026-11-15, before any new construction method was screened.
These rules are not changed after results are seen. Any change must be
committed, with its reason, before the stage it affects is evaluated.

## Scope

Memory construction only. Encoder, classifier head, retrieval rule and
beta-selection procedure are fixed. Phase 1 uses training and validation
data only (beta and method hyperparameters on the validation-tune half,
metrics on the validation-eval half). The test split is not read.

## Primary metric and comparator

- Primary metric: noise AUC = mean of the Gaussian, salt-and-pepper and
  occlusion robustness AUCs on the calibrated grids (configs/cifar10.yaml).
- Comparator: class-wise k-means at the SAME budget k (patterns per class),
  paired by seed. Budgets: k in {2, 4, 8, 16}. Seeds: 0-9.
- "pp" = percentage points (absolute difference in accuracy, x100).

## Bands (per method)

A method is placed in a band using its paired mean gain in noise AUC over
k-means. Every band above Fail also requires ALL of:
  (a) at least 7 of 10 seeds positive,
  (b) the band reached at at least 2 of the 4 budgets,
  (c) clean accuracy not lower than k-means by more than 0.3 pp at those budgets.

| Band          | Mean gain vs k-means | Consequence                                   |
|---------------|----------------------|-----------------------------------------------|
| Pass          | >= +0.3 pp           | Eligible as a Phase 2 finalist                |
| Weak positive | +0.1 to +0.3 pp      | Not a finalist alone; reported as small and   |
|               |                      | unconfirmed; may enter the combination stage  |
| Fail          | < +0.1 pp, or (a)-(c) not met | Stop; record result and reason        |

Rationale: the SD of paired differences in noise AUC is about 0.29 pp
(CIFAR-10 screening), so with 10 seeds the mean wobbles by about 0.09 pp
by chance. A useless method exceeds +0.1 pp by luck about 14% of the time
and +0.3 pp about 0.06% of the time. +0.3 pp is also about the gain from
4 to 8 k-means patterns per class and is confirmable with about 14-20 seeds.

## Combination stage (proposal Section 5.5)

- Run only if at least two of boundary-weighted, rarity-weighted and
  noise-aware FL reached Pass or Weak positive.
- Boundary + rarity are combined only if their within-class Spearman rank
  correlation (B vs 1/rho) is below 0.7; otherwise keep the better one.
- The combination must itself reach Pass to become a finalist.

## Finalists

At most 2 methods in the Pass band, ranked by mean gain at k = 8;
ties broken by lower construction time. If none pass, Phase 2 is not run
and the negative result is reported.

## Stage 1 diagnostic rules (decided before running the diagnostics)

- Classification-aware accuracy objective (CA-acc) is dropped if training
  nearest-pattern accuracy of the k-means memory (k = 8) exceeds 0.99.
- Noise-aware FL is deprioritised (run last among coverage approaches) if
  corrupted validation queries already have a same-class training
  embedding among their 20 nearest clean neighbours at least as often as
  their clean versions do.

## No re-tuning after failure

A method that fails is not re-tuned on the same validation data.
Hyperparameter grids are those stated in the proposal (v1.1, Chapter 5).

## Reference

Full methodology: "Improving Memory Construction in HopfieldPC Through
Weighted Facility Location, Classification Awareness, and Diversity-Based
Selection", research proposal, version 1.1 (2026-11-15), Yibeltal Marie.
Link: https://drive.google.com/file/d/1D-RjCMVkYPWutMEmxCk40-Y0DcHtaIvm/view?usp=sharing

These criteria refer to that version. If a later version of the proposal
differs, this file takes precedence for Phase 1 decisions until it is
explicitly amended by a new commit.
