# Experiment log — `experiment/diversity-memory`

Entries are dated and factual. Observations, inferences and open questions are labelled as such.
Checkpoints, datasets and generated reports are stored outside the repository and are not committed.

---

## 2026-11-03 — G0: environment and baseline reproduction

### Repository state

| Item | Value |
|---|---|
| Branch | `experiment/diversity-memory` |
| Commit | `1f33022` (same as `upstream/experiment-id1`) |
| Working tree | Clean |

### Environment

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | Tesla T4 |
| Python | 3.13.16 |
| torch | 2.11.0+cu130 |

### Commands

Config: `configs/mnist_prototype_retrieval.yaml` (unchanged). `<DATA_DIR>` and `<REPORTS_DIR>` are local paths outside the repository.

```bash
pip install -r requirements.txt
pytest -q

# Smoke run (1 seed, small data, 1 epoch)
python -m benchmarks.mnist_prototype_retrieval --quick \
    --data-dir <DATA_DIR> --reports-dir <REPORTS_DIR>/quick

# Single full seed
python -m benchmarks.mnist_prototype_retrieval --seed 0 \
    --data-dir <DATA_DIR> --reports-dir <REPORTS_DIR>/full

# Full run, all seeds from the config (10)
python -m benchmarks.mnist_prototype_retrieval \
    --data-dir <DATA_DIR> --reports-dir <REPORTS_DIR>/full
```

### Results

- `pytest -q`: 353 passed in 14.22 s.
- Quick run completed. S4 fails as expected for quick mode (2,000 training images, 1 epoch); S11 fails.
- Baseline tuning selected lr = 0.001, batch = 128, epochs = 6 (validation clean accuracy 0.9873).
- Full 10-seed run: 3 min 46 s (tuning results and seed-0 checkpoint cached). Selected β* = 100.

Test split, 10 seeds, Hopfield memory = class-wise K-means, 8 per class:

| Metric | Baseline | Hopfield | Δ | Seeds +/− | Wilcoxon p |
|---|---|---|---|---|---|
| Clean accuracy | 0.9898 ± 0.0008 | 0.9902 ± 0.0007 | +0.04 pp | 6/3 | 0.238 |
| Gaussian AUC | 0.6230 | 0.6512 | +2.82 pp | 8/2 | 0.010 |
| Salt-and-pepper AUC | 0.7323 | 0.7635 | +3.12 pp | 8/2 | 0.010 |
| Occlusion AUC | 0.8745 | 0.8808 | +0.62 pp | 8/2 | 0.027 |

Memory-design ablation, Gaussian σ = 0.5, 10-seed mean accuracy:

| Memory | Accuracy |
|---|---|
| Baseline (no memory) | 0.6046 |
| 1 raw mean per class | 0.6354 |
| 1 centred mean per class | 0.6622 |
| 8 centred K-means per class | 0.6667 |

- Sanity checks S1–S10 pass. S11 (latency overhead < 10 %) fails at +28.7 % at batch 256.

### Observations

- Seed 0 alone shows the opposite sign (Gaussian −1.44 pp, salt-and-pepper −0.59 pp); it is one of the two negative seeds.
- Seed-to-seed variance is large (baseline accuracy at σ = 0.5: 0.605 ± 0.092), so only paired per-seed comparisons on the same backbone are meaningful.
- Going from 10 centred means to 80 K-means patterns adds about +0.45 pp at σ = 0.5. Effects of a different memory at 8 per class are expected to be small.
- The README reports +7.2 pp (Gaussian) and +7.8 pp (salt-and-pepper). This run measured +2.82 pp and +3.12 pp: same direction, significant, smaller magnitude.

### Open questions

- Reason for the difference from the README figures (to discuss with the project leader).

### Next

- Implement the class-wise facility-location selector (P5) and its unit tests.
- Test H9 (low budgets, 2–4 per class) on validation data first.

---

## 2026-11-04 — P5: real-embedding selection implemented; seed-0 diagnostic

- Added `benchmarks/memory_selection.py` (class-wise greedy facility location on train-mean-centred
  cosine; class-wise random control) and `build_memory(method=...)`; default `kmeans` unchanged.
- `pytest -q`: 17 new tests pass; full suite 370 passed in 17.01 s.
- Diagnostic, seed 0, training and validation data only (no test data). Coverage = q from each
  validation embedding to its nearest same-class stored pattern.

| Method | k/class | Build (s) | Head correct on stored | Val mean q | Val min q |
|---|---|---|---|---|---|
| kmeans | 2 | 0.45 | 20/20 | 0.9593 | 0.4929 |
| facility_location | 2 | 12.94 | 20/20 | 0.9552 | 0.4626 |
| random | 2 | 0.01 | 20/20 | 0.9347 | 0.4438 |
| kmeans | 4 | 0.52 | 40/40 | 0.9663 | 0.5056 |
| facility_location | 4 | 17.10 | 40/40 | 0.9638 | 0.5424 |
| random | 4 | 0.01 | 40/40 | 0.9471 | 0.5236 |
| kmeans | 8 | 0.64 | 80/80 | 0.9718 | 0.5450 |
| facility_location | 8 | 25.41 | 80/80 | 0.9697 | 0.5424 |
| random | 8 | 0.01 | 80/80 | 0.9604 | 0.5236 |

### Observations (single seed; not a result)

- Facility location covers validation data better than random at every budget (+0.009 to +0.021),
  and slightly worse than k-means (-0.002 to -0.004); the gap to k-means is largest at k=2.
- The head classifies every stored pattern correctly for all methods: the H5 failure mode was not
  observed on seed 0, so the head-agreement part of C2 currently has no failure to fix.
- Facility-location and random selections are nested across budgets, so the worst-case value can
  repeat between k=4 and k=8; the min is driven by a single point and is descriptive only.
- Facility location costs 13-25 s per seed versus about 0.5 s for k-means.

### Next

- Comparison script: every method and budget on the same cached backbones, beta selected on
  validation per method, Gaussian and salt-and-pepper AUC over 10 seeds; head agreement on all seeds.

---

## 2026-11-04 — RQ1 result: memory selection benchmark (10 seeds)

- Command: `python -m benchmarks.memory_selection_benchmark --reports-dir <REPORTS_DIR>`
  (cached backbones for all 10 seeds; wall time 2 min 16 s). Tests: 382 passed.
- Protocol frozen before test evaluation (G4): primary = Gaussian and salt-and-pepper AUC vs k-means
  at equal budget; Holm over six primary tests per method; clean-accuracy non-inferiority 0.005;
  beta tuned identically per design on validation only.
- beta*: k-means and facility location 100 at every budget; class means 100; random 10 / 20 / 20.

| k/class | FL − k-means, Gaussian AUC | FL − k-means, S&P AUC | Random − k-means, S&P AUC |
|---|---|---|---|
| 2 | −0.45 pp (0/10 seeds +; Holm p 0.012) LOSS | −0.45 pp (1/10; 0.039) LOSS | −2.17 pp (0/10) LOSS |
| 4 | −0.14 pp (4/10; 0.75) n.s. | −0.35 pp (1/10; 0.029) LOSS | −1.90 pp (0/10) LOSS |
| 8 | −0.00 pp (5/10; 0.92) n.s. | −0.22 pp (4/10; 0.25) n.s. | −1.18 pp (0/10) LOSS |

- Clean accuracy: facility location non-inferior at every budget; random inferior at k=2.
- Head correct on stored patterns: facility location 800/800 (k=8, all seeds); random 795/800.
- Validation coverage (k=8): k-means 0.9715, facility location 0.9696, random 0.9602.

### Conclusions

- RQ1: facility location does not beat k-means at any budget; it loses at low budgets and ties at
  k=8. Ordering at every budget: k-means >= facility location > random.
- H9 refuted: the gap to k-means is largest at k=2 and closes by k=8.
- H5 failure mode not observed for facility location, so C2 is not added (no failure to fix).
  The construction method stays C1 only.
- Interpretation (not tested): centroid averaging acts as denoising; random memories selecting a
  lower beta on validation is consistent with this.
- Effect sizes are small (< 0.5 pp) relative to the memory's gain over no memory (+2.8 pp).

### Next

- RQ2: bounded maintenance with facility-location marginal-gain replacement.
- Backlog (needs leader approval, validation-only selection, labelled post hoc): hybrid that selects
  regions by facility location and stores local averages.

---

## 2026-11-04 — P6b diagnostics (validation only, 10 seeds)

- Command: `python -m benchmarks.memory_diagnostics --reports-dir <REPORTS_DIR>` (1 min 46 s). Tests: 390 passed.
- Validation split per seed into tuning (beta) and evaluation halves (2,500 each); beta grid extended to 1e4.

### Results

- beta* = 1e4 (nearest pattern) for every averaged and facility-location design (k-means k=8: 1000); random 10-20.
  Validation curves are flat above beta = 100 (< 0.1 pp), so the original grid ceiling is immaterial.
- N_eff = 1.00 and blend error = 0.0000 for all non-random designs: retrieval acts as a nearest-pattern classifier
  and every error is a retrieval error (nearest stored pattern of the wrong class). Hypothesis C ruled out.
- Audit C (robust acc = mean over Gaussian 0.3/0.5, S&P 0.1/0.2, occlusion 10; paired by seed; p descriptive):
  - k=2: averaging | FL regions +0.36 pp (8/1, p 0.008); averaging | k-means regions +0.07 (6/4);
    regions | averages -0.14 (3/7); regions | real points -0.42 (2/8, p 0.02).
  - k=8: averaging | FL regions +0.22 (8/2, p 0.014); averaging | k-means regions +0.13 (8/2, p 0.047);
    regions | averages +0.05 (6/4); regions | real points -0.04 (5/5).
- FL picks lie close to their region means (cosine 0.993-0.995). Centred norm: picks about 34-35, averages about 29.
- B3: all designs lose coverage at similar rates under noise; cos(noisy, clean) = 0.61 at Gaussian 0.5.
- B4: boundary/rare robust accuracy about 0.57 vs typical 0.86 for all designs; between-design differences <= 1 pp.
- All non-random designs lie within 0.5 pp robust accuracy; seed SD 3.4 pp.

### Conclusions

- The facility-location loss is explained by storing real points instead of averages; region choice contributes
  little (only at k=2 with real points). FL regions + averaging ties k-means (+0.05 pp at k=8).
- Construction choice has limited downstream impact on MNIST with this encoder, provided patterns are centred
  averages. M2 (difficulty weighting), M3 (retrieval-aware) and M4 (corruption-aware) are not motivated.
- G5b (proposed, pending leader): close RQ1 as a negative result with an identified mechanism. For RQ2, consider
  facility-location gain for admission with running-average pattern updates (deviation from P3, needs approval).
