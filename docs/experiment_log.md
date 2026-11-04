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
