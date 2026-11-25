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


## Amendment 1 (2026-11-16): rarity score for rarity-weighted FL

Reason: the Stage 1 diagnostics (CIFAR-10, 10 seeds) showed the density
proxy rho is compressed into about [0.98, 0.998] (median 0.988). The
proposal v1.1 weight (rho + eps)^(-alpha) with alpha <= 2 then varies by
only about 4% across a class, i.e. it is effectively unweighted.

Change, made before any rarity-weighted result exists:
- Rarity score R_i = within-class rank of 1/rho_i scaled to [0, 1]
  (average ranks for ties; most crowded = 0, rarest = 1).
- Weight w_i = 1 + lambda_r * R_i, lambda_r in {0.5, 1, 2, 4}.
- Label-consistency filter unchanged: R_i = 0 if B_i > tau_B.
- Percentile clipping is no longer needed (ranks are bounded); the
  density proxy, kappa grid and all other rules are unchanged.
- Weights stay fixed and positive, so weighted FL keeps its (1 - 1/e)
  guarantee for its own objective. The same R_i is used in the
  combination stage, as in proposal Section 5.5.


## Amendment 2 (2026-11-24): primary metric as in the research plan; clean margin 1 pp (provisional)

Decided by Yibeltal Marie on 2026-11-24, pending confirmation by the project leader.
Applies to every method screened after this commit. Stages 2-4 keep the decisions made
under the original rule (all Fail); their code paths are unchanged.

Changes
1. Primary robustness metric: known-type AUC = mean of the Gaussian and salt-and-pepper
   AUCs, as in the research plan (P4.3). Occlusion AUC and the repository's three-type
   noise AUC become secondary and are always reported.
2. Clean guard: mean clean-accuracy loss vs class-wise k-means at most 1.0 pp (was 0.3 pp).
   The plan's provisional margin is 0.5 pp; 1.0 pp is looser and is set "for now". It may be
   revised (e.g. after the leader's confirmation), but only by a further dated amendment
   committed before the next screening run, never in response to a candidate's results.
Unchanged: +0.3 pp (pass) / +0.1 pp (weak) bands on the primary metric, at least 70 % of
seeds positive, at least 2 budgets, at most 2 finalists, plain k-means as reference,
tune-half selection, test split untouched until Phase 2.

Disclosure (post hoc; not decisions)
This amendment was made after the Stage 3 and Stage 4 validation results were known.
Re-reading those results under it:
- Stage 3 variant A (gamma 0.25): Pass. k = 2: known AUC +0.37 [-0.16, +0.90] (7/10),
  clean -0.64; k = 8: +0.60 [-0.23, +1.42] (7/10), clean -0.85; k = 4 fails (clean -1.17);
  k = 16 fails (6/10). Fragile: both passing budgets sit exactly at 7/10, both CIs include
  0, occlusion AUC -0.6 to -1.9 (0-1/10). Under the plan's 0.5 pp it would remain Fail.
- Stage 4 KV-M: Fail (clean -1.63 to -2.30 pp) despite known AUC +2.9 to +10.1 (9-10/10).
Because the rule was set after these results, a method that passes only under it,
including Stage 3 variant A, is a post-hoc candidate: it must be confirmed under a
pre-registration on data not used here (replicate backbone set, MNIST Phase 1, Phase 2
test) before any claim of improvement or any PR proposing it.
