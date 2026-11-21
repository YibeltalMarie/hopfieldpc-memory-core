"""Stage 3: noise-aware facility location (validation only; foundation proposal Chapter 5).

Every training image is encoded twice: clean (z_i) and once corrupted (z~_i; Gaussian
or salt-and-pepper, level uniform on [0, largest grid level], training noise stream 2;
occlusion is never used). Class-wise facility location then covers both:

    F_c(S) = sum_i q(z_i, s(i)) + gamma * sum_i q(z~_i, s(i)),   s(i) = best pattern in S

Variants (gamma in GAMMAS, budgets k in BUDGETS):

    na_fl_A   candidates = clean embeddings of the class    (stores real clean embeddings)
    na_fl_B   candidates = clean + corrupted embeddings      (may store corrupted embeddings)
    na_fl_C   variant A's picks replaced by region means over clean + corrupted points
    na_kmeans repository k-means on clean + corrupted embeddings (attribution control)

plus kmeans, facility_location and no_memory, all under memory_screening's protocol
(beta on the validation-tune half, metrics on the evaluation half, paired by seed,
test split never read) with the descriptive training-split column.

Pre-registered selection (tune half only): for each variant and k, gamma = argmax of the
tune score at the design's own beta* (ties -> smaller gamma); the candidate variant is
the one with the best tune score averaged over budgets. Only the candidate is judged by
the screening criteria, against plain kmeans at the same k; everything else is descriptive.

    python -m benchmarks.noise_aware_screening --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks import memory_screening as ms
from benchmarks.construction_diagnostics import corrupted_train_embeddings
from benchmarks.memory_diagnostics import paired_summary
from benchmarks.memory_diagnostics import region_means as cosine_region_means
from benchmarks.memory_selection import nested_prefix, noise_aware_selection, select_class_memory
from benchmarks.mnist_baseline import mean_std, noise_grids
from benchmarks.mnist_prototype_retrieval import build_memory, select_beta
from benchmarks.runs import build_arg_parser, config_from_args
from benchmarks.screening_criteria import band_at_budget, method_band

BUDGETS = ms.BUDGETS
GAMMAS = (0.25, 0.5, 1.0)
VARIANTS = ("A", "B", "C")
COPY_TYPES = ("gaussian", "salt_pepper")
REFERENCE = "kmeans"
KNOWN_TYPES = ("gaussian", "salt_pepper")
TRANSFER_TYPE = "occlusion"


def na_name(variant: str, gamma: float, k: int) -> str:
    return f"na_fl_{variant}_g{gamma:g}_k{k}"


def design_names() -> list[str]:
    names = ["no_memory"] + [f"{base}_k{k}" for base in ("kmeans", "facility_location", "na_kmeans")
                             for k in BUDGETS]
    return names + [na_name(v, g, k) for v in VARIANTS for g in GAMMAS for k in BUDGETS]


def make_builder(grids: dict[str, list[float]]):
    """Design builder for memory_screening.run_screening (needs the calibrated grid maxima)."""
    max_level = {t: max(grids[t]) for t in COPY_TYPES}

    def build(run: Any, device: torch.device) -> dict[str, dict[str, Any]]:
        z, y = run.z_train, run.splits.train_labels
        center = z.mean(dim=0)                       # clean training mean, as retrieval uses
        designs: dict[str, dict[str, Any]] = {}

        def add(name: str, memory: torch.Tensor, labels: torch.Tensor, secs: float) -> None:
            designs[name] = {"memory": memory.cpu(), "labels": labels.cpu(), "center": center, "build_s": secs}

        start = time.perf_counter()
        z_noisy = corrupted_train_embeddings(run, max_level)   # stream 2, one copy per image, row-aligned
        copy_s = time.perf_counter() - start
        z_union, y_union = torch.cat([z, z_noisy]), torch.cat([y, y])
        k_max = max(BUDGETS)
        classes = torch.unique(y).tolist()
        for k in BUDGETS:
            for base, zz, yy, extra in (("kmeans", z, y, 0.0), ("na_kmeans", z_union, y_union, copy_s)):
                start = time.perf_counter()
                memory, labels, _ = build_memory(zz, yy, per_class=k, center=True, seed=run.seed)
                add(f"{base}_k{k}", memory, labels, extra + time.perf_counter() - start)
        start = time.perf_counter()   # standard FL exactly as in the earlier screenings
        sel, sel_labels, _ = select_class_memory(z.to(device), y.to(device), per_class=k_max,
                                                 method="facility_location", center=center.to(device), seed=run.seed)
        fl_s = time.perf_counter() - start
        for k in BUDGETS:
            memory, labels = nested_prefix(sel.cpu(), sel_labels.cpu(), per_class_max=k_max, per_class=k)
            add(f"facility_location_k{k}", memory, labels, fl_s)
        for gamma in GAMMAS:
            for variant, allow_noisy in (("A", False), ("B", True)):
                start = time.perf_counter()
                picks = {c: noise_aware_selection(z[y == c], z_noisy[y == c], k_max, gamma, center,
                                                  allow_noisy=allow_noisy, device=device) for c in classes}
                secs = copy_s + time.perf_counter() - start
                for k in BUDGETS:
                    memory = torch.cat([picks[c][:k] for c in classes])
                    labels = torch.tensor(classes).repeat_interleave(k)
                    add(na_name(variant, gamma, k), memory, labels, secs)
                    if variant == "A":
                        start_c = time.perf_counter()
                        means, _ = cosine_region_means(z_union, y_union, memory, labels, center)
                        add(na_name("C", gamma, k), means, labels, secs + time.perf_counter() - start_c)
        return designs

    return build


def run_noise_aware(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    raw = ms.run_screening(cfg, retrain=retrain, builder=make_builder(noise_grids(cfg)), names=design_names(),
                           train_eval=True)
    raw["selection"] = select_candidate(raw)
    raw["decision"] = candidate_decision(raw)
    return raw


def tune_score(raw: dict[str, Any], name: str) -> float:
    """Seed-mean validation-tune score at the design's own beta* (the beta-selection score)."""
    _, mean_curve = select_beta(raw["beta_grid"], raw["beta_curves"][name])
    return max(mean_curve)


def select_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered, tune-half-only choice of gamma per (variant, k) and of the candidate variant."""
    gamma = {v: {} for v in VARIANTS}
    score = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for k in BUDGETS:
            scores = [tune_score(raw, na_name(v, g, k)) for g in GAMMAS]
            best = max(scores)
            gamma[v][str(k)] = GAMMAS[scores.index(best)]    # first maximum -> smaller gamma on ties
            score[v][str(k)] = best
    mean_score = {v: sum(score[v].values()) / len(BUDGETS) for v in VARIANTS}
    candidate = max(VARIANTS, key=lambda v: (mean_score[v], -VARIANTS.index(v)))
    return {"gamma": gamma, "tune_score": score, "mean_tune_score": mean_score, "candidate": candidate}


def chosen(raw: dict[str, Any], variant: str, k: int) -> str:
    return na_name(variant, raw["selection"]["gamma"][variant][str(k)], k)


def paired(results: dict[str, Any], a: str, b: str, metric: str) -> dict[str, Any]:
    return paired_summary([ra[metric] - rb[metric] for ra, rb in zip(results[a], results[b])])


def known_auc(row: dict[str, Any]) -> float:
    return sum(row[f"{t}_auc"] for t in KNOWN_TYPES) / len(KNOWN_TYPES)


def candidate_decision(raw: dict[str, Any]) -> dict[str, Any]:
    v = raw["selection"]["candidate"]
    per_k = {}
    for k in BUDGETS:
        a, b = chosen(raw, v, k), f"{REFERENCE}_k{k}"
        noise, clean = paired(raw["results"], a, b, "noise_auc"), paired(raw["results"], a, b, "clean")
        per_k[str(k)] = {"design": a, "noise": noise, "clean": clean, "band": band_at_budget(noise, clean)}
    return {"candidate": v, "per_budget": per_k, "band": method_band({k: x["band"] for k, x in per_k.items()})}


def _diff(s: dict[str, Any]) -> str:
    return (f"{100 * s['mean']:+.2f} [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}] "
            f"({s['n_pos']}/{s['n_neg']})")


def build_report(raw: dict[str, Any]) -> str:
    sel, dec = raw["selection"], raw["decision"]
    res = raw["results"]
    lines = [f"# Stage 3: noise-aware facility location — {raw['dataset']} (validation only)", "",
             f"Seeds {raw['seeds']}; budgets k = {list(BUDGETS)}; gamma in {list(GAMMAS)}; one corrupted copy per "
             "training image (Gaussian or salt-and-pepper, training noise stream 2; no occlusion). beta per design "
             "on the tuning half; metrics on the evaluation half; differences in pp, mean [95% t-CI] (seeds +/-). "
             "The test split was not read.", "",
             "## 1. Tune-half selection (fixed before evaluation metrics)", "",
             "| Variant | mean tune score | " + " | ".join(f"gamma at k = {k}" for k in BUDGETS) + " |",
             "|---|---|" + "---|" * len(BUDGETS)]
    for v in VARIANTS:
        lines.append(f"| {v} | {sel['mean_tune_score'][v]:.4f} | "
                     + " | ".join(f"{sel['gamma'][v][str(k)]:g}" for k in BUDGETS) + " |")
    lines += ["", f"Candidate variant: **{sel['candidate']}**", "",
              f"## 2. Decision (docs/phase1_criteria.md): variant {dec['candidate']} vs kmeans at equal k", "",
              f"**Band: {dec['band']}**", "", "| k | design | noise AUC diff | clean diff | band at k |",
              "|---|---|---|---|---|"]
    for k, x in dec["per_budget"].items():
        lines.append(f"| {k} | {x['design']} | {_diff(x['noise'])} | {_diff(x['clean'])} | {x['band']} |")
    lines += ["", "## 3. Descriptive comparisons (noise AUC unless stated; each variant at its tune-selected gamma)",
              "", "| Comparison | " + " | ".join(f"k = {k}" for k in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    rows = []
    for v in VARIANTS:
        rows.append((f"Variant {v} - kmeans", lambda k, v=v: paired(res, chosen(raw, v, k), f"kmeans_k{k}",
                                                                    "noise_auc")))
        rows.append((f"Variant {v} - na_kmeans (attribution)",
                     lambda k, v=v: paired(res, chosen(raw, v, k), f"na_kmeans_k{k}", "noise_auc")))
    rows.append(("na_kmeans - kmeans", lambda k: paired(res, f"na_kmeans_k{k}", f"kmeans_k{k}", "noise_auc")))
    rows.append(("facility_location - kmeans",
                 lambda k: paired(res, f"facility_location_k{k}", f"kmeans_k{k}", "noise_auc")))
    v = dec["candidate"]
    for t in KNOWN_TYPES + (TRANSFER_TYPE,):
        rows.append((f"Variant {v} - kmeans, {t} AUC",
                     lambda k, t=t: paired(res, chosen(raw, v, k), f"kmeans_k{k}", f"{t}_auc")))
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(_diff(fn(k)) for k in BUDGETS) + " |")

    gaps_noise, gaps_clean = ms.memory_gaps(raw, "noise_auc"), ms.memory_gaps(raw, "clean")
    lines += ["", "## 4. Training / validation record (descriptive; training queries: 5000 per seed, noise stream 3, "
              "beta from validation)", "",
              "| Design | train clean | val clean | train noise AUC | val noise AUC | memory gap, noise AUC (pp) | "
              "memory gap, clean (pp) | queries stored |", "|---|---|---|---|---|---|---|---|"]
    shown = ["no_memory"] + [f"{b}_k{k}" for b in ("kmeans", "facility_location", "na_kmeans") for k in BUDGETS]
    shown += [chosen(raw, vv, k) for vv in VARIANTS for k in BUDGETS]
    for name in shown:
        tr, va = raw["train_results"][name], res[name]
        stored = raw["train_stored_fraction"].get(name)
        lines.append(
            f"| {name} | {mean_std([r['clean'] for r in tr])[0]:.4f} | {mean_std([r['clean'] for r in va])[0]:.4f} | "
            f"{mean_std([r['noise_auc'] for r in tr])[0]:.4f} | {mean_std([r['noise_auc'] for r in va])[0]:.4f} | "
            f"{'—' if name == 'no_memory' else _diff(gaps_noise[name])} | "
            f"{'—' if name == 'no_memory' else _diff(gaps_clean[name])} | "
            f"{'—' if stored is None else f'{mean_std(stored)[0]:.4f}'} |")
    lines += ["", "## 5. Absolute validation values (mean ± std over seeds)", "",
              "| Design | clean | noise AUC | known-type AUC | occlusion AUC | beta* |", "|---|---|---|---|---|---|"]
    for name in design_names():
        rr = res[name]
        beta = raw["beta_star"].get(name)
        c, n = mean_std([r["clean"] for r in rr]), mean_std([r["noise_auc"] for r in rr])
        kn, oc = mean_std([known_auc(r) for r in rr]), mean_std([r[f"{TRANSFER_TYPE}_auc"] for r in rr])
        lines.append(f"| {name} | {c[0]:.4f} ± {c[1]:.4f} | {n[0]:.4f} ± {n[1]:.4f} | {kn[0]:.4f} | {oc[0]:.4f} | "
                     f"{'—' if beta is None else f'{beta:g}'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("Stage 3: noise-aware facility location screening (validation only).").parse_args(argv)
    cfg = config_from_args(args)
    raw = run_noise_aware(cfg, retrain=args.retrain)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "noise_aware_screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
