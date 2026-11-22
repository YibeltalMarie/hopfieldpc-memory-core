"""Stage 4: key-value noise-aware facility location (validation only; foundation proposal Section 5.3).

Stage 3 showed that a stored pattern is both matched (key) and returned (value), so a
stored corrupted embedding returns a corrupted vector. Here keys and values are separated:

    keys     chosen as Stage 3 variant B: class-wise facility location over clean + corrupted
             embeddings (weights 1 and gamma), clean or corrupted candidates
    values   clean:  KV-S  the clean embedding of the key's source image
                     KV-M  the weighted mean clean source embedding of the points the key serves
    readout  h(z) = sum_j softmax_j(beta * cos(z - mu, k_j - mu)) v_j   (memory_screening.KeyValueBridge)

The corrupted copies are Stage 3's (one per training image, Gaussian or salt-and-pepper,
level uniform on [0, largest grid level], training noise stream 2; occlusion never used).
Designs (gamma in GAMMAS, budgets k in BUDGETS):

    kv_S, kv_M           the two key-value variants
    na_fl_B              the same keys returning themselves (mechanism check: kv_S - na_fl_B)
    na_fl_A              Stage 3 variant A (clean candidates)
    kmeans, facility_location, na_kmeans, no_memory

all under memory_screening's protocol (beta on the validation-tune half, metrics on the
evaluation half, paired by seed, test split never read) with the training-split column.

Pre-registered selection (tune half only): for kv_S and kv_M and each k, gamma = argmax
of the tune score at the design's beta* (ties -> smaller gamma); the candidate is the
variant with the best tune score averaged over budgets (ties -> kv_S). Only the candidate
is judged by the screening criteria, against plain kmeans at the same k. Comparisons with
na_fl_A and na_fl_B use the candidate's gamma at each k; everything except the decision is
descriptive.

    python -m benchmarks.key_value_screening --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks import memory_screening as ms
from benchmarks import noise_aware_screening as na
from benchmarks.construction_diagnostics import corrupted_train_embeddings
from benchmarks.memory_diagnostics import paired_summary
from benchmarks.memory_selection import (key_value_selection, nested_prefix, noise_aware_selection,
                                         select_class_memory, served_clean_means)
from benchmarks.mnist_baseline import mean_std, noise_grids
from benchmarks.mnist_prototype_retrieval import build_memory
from benchmarks.runs import build_arg_parser, config_from_args
from benchmarks.screening_criteria import band_at_budget, method_band

BUDGETS = ms.BUDGETS
GAMMAS = (0.25, 0.5, 1.0, 2.0)
KV_VARIANTS = ("S", "M")
REFERENCE = "kmeans"
PER_LEVEL_K = 8   # budget of the per-level table (repository default)


def kv_name(variant: str, gamma: float, k: int) -> str:
    return f"kv_{variant}_g{gamma:g}_k{k}"


def design_names() -> list[str]:
    names = ["no_memory"] + [f"{base}_k{k}" for base in ("kmeans", "facility_location", "na_kmeans")
                             for k in BUDGETS]
    names += [na.na_name(v, g, k) for v in ("A", "B") for g in GAMMAS for k in BUDGETS]
    return names + [kv_name(v, g, k) for v in KV_VARIANTS for g in GAMMAS for k in BUDGETS]


def make_builder(grids: dict[str, list[float]]):
    """Design builder for memory_screening.run_screening (needs the calibrated grid maxima)."""
    max_level = {t: max(grids[t]) for t in na.COPY_TYPES}

    def build(run: Any, device: torch.device) -> dict[str, dict[str, Any]]:
        z, y = run.z_train, run.splits.train_labels
        center = z.mean(dim=0)                       # clean training mean, as retrieval uses
        designs: dict[str, dict[str, Any]] = {}

        def add(name: str, memory: torch.Tensor, labels: torch.Tensor, secs: float, **extra: Any) -> None:
            designs[name] = {"memory": memory.cpu(), "labels": labels.cpu(), "center": center, "build_s": secs,
                             **extra}

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
            start = time.perf_counter()
            picks_a = {c: noise_aware_selection(z[y == c], z_noisy[y == c], k_max, gamma, center,
                                                allow_noisy=False, device=device) for c in classes}
            a_s = copy_s + time.perf_counter() - start
            start = time.perf_counter()
            kv = {c: key_value_selection(z[y == c], z_noisy[y == c], k_max, gamma, center, device=device)
                  for c in classes}
            kv_s = copy_s + time.perf_counter() - start
            for k in BUDGETS:
                labels = torch.tensor(classes).repeat_interleave(k)
                keys = torch.cat([kv[c][0][:k] for c in classes])
                values_s = torch.cat([kv[c][1][:k] for c in classes])
                info = {"noisy_keys": torch.cat([kv[c][2][:k] for c in classes]).float().mean().item()}
                add(na.na_name("A", gamma, k), torch.cat([picks_a[c][:k] for c in classes]), labels, a_s)
                add(na.na_name("B", gamma, k), keys, labels, kv_s, info=info)
                add(kv_name("S", gamma, k), keys, labels, kv_s, values=values_s, info=info)
                start = time.perf_counter()
                values_m = torch.cat([served_clean_means(z[y == c], z_noisy[y == c], kv[c][0][:k], gamma, center,
                                                         fallback=kv[c][1][:k], device=device) for c in classes])
                add(kv_name("M", gamma, k), keys, labels, kv_s + time.perf_counter() - start, values=values_m,
                    info=info)
        return designs

    return build


def run_key_value(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    raw = ms.run_screening(cfg, retrain=retrain, builder=make_builder(noise_grids(cfg)), names=design_names(),
                           train_eval=True)
    raw["selection"] = select_candidate(raw)
    raw["decision"] = candidate_decision(raw)
    return raw


def select_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered, tune-half-only choice of gamma per (variant, k) and of the candidate variant."""
    gamma = {v: {} for v in KV_VARIANTS}
    score = {v: {} for v in KV_VARIANTS}
    for v in KV_VARIANTS:
        for k in BUDGETS:
            scores = [na.tune_score(raw, kv_name(v, g, k)) for g in GAMMAS]
            best = max(scores)
            gamma[v][str(k)] = GAMMAS[scores.index(best)]    # first maximum -> smaller gamma on ties
            score[v][str(k)] = best
    mean_score = {v: sum(score[v].values()) / len(BUDGETS) for v in KV_VARIANTS}
    candidate = max(KV_VARIANTS, key=lambda v: (mean_score[v], -KV_VARIANTS.index(v)))
    return {"gamma": gamma, "tune_score": score, "mean_tune_score": mean_score, "candidate": candidate}


def chosen(raw: dict[str, Any], variant: str, k: int) -> str:
    return kv_name(variant, raw["selection"]["gamma"][variant][str(k)], k)


def at_candidate_gamma(raw: dict[str, Any], variant: str, k: int) -> str:
    """Stage 3 design (na_fl_A or na_fl_B) at the candidate's gamma for budget k."""
    cand = raw["selection"]["candidate"]
    return na.na_name(variant, raw["selection"]["gamma"][cand][str(k)], k)


def candidate_decision(raw: dict[str, Any]) -> dict[str, Any]:
    v = raw["selection"]["candidate"]
    per_k = {}
    for k in BUDGETS:
        a, b = chosen(raw, v, k), f"{REFERENCE}_k{k}"
        noise, clean = na.paired(raw["results"], a, b, "noise_auc"), na.paired(raw["results"], a, b, "clean")
        per_k[str(k)] = {"design": a, "noise": noise, "clean": clean, "band": band_at_budget(noise, clean)}
    return {"candidate": v, "per_budget": per_k, "band": method_band({k: x["band"] for k, x in per_k.items()})}


def level_diff(raw: dict[str, Any], a: str, b: str, kind: str, i: int) -> dict[str, Any]:
    """Paired accuracy difference at level index i of noise type ``kind`` (evaluation half)."""
    pairs = zip(raw["results"][a], raw["results"][b])
    return paired_summary([ra[f"{kind}_acc"][i] - rb[f"{kind}_acc"][i] for ra, rb in pairs])


def build_report(raw: dict[str, Any]) -> str:
    sel, dec = raw["selection"], raw["decision"]
    res, info = raw["results"], raw.get("design_info", {})
    v = dec["candidate"]
    lines = [f"# Stage 4: key-value noise-aware facility location — {raw['dataset']} (validation only)", "",
             f"Seeds {raw['seeds']}; budgets k = {list(BUDGETS)}; gamma in {list(GAMMAS)}; keys = Stage 3 variant B "
             "selection (clean + corrupted candidates); values clean (KV-S source image, KV-M served mean). Same "
             "corrupted copies as Stage 3 (training noise stream 2; no occlusion). beta per design on the tuning "
             "half; metrics on the evaluation half; differences in pp, mean [95% t-CI] (seeds +/-). The test split "
             "was not read.", "",
             "## 1. Tune-half selection (fixed before evaluation metrics)", "",
             "| Variant | mean tune score | " + " | ".join(f"gamma at k = {k}" for k in BUDGETS) + " |",
             "|---|---|" + "---|" * len(BUDGETS)]
    for vv in KV_VARIANTS:
        lines.append(f"| KV-{vv} | {sel['mean_tune_score'][vv]:.4f} | "
                     + " | ".join(f"{sel['gamma'][vv][str(k)]:g}" for k in BUDGETS) + " |")
    lines += ["", f"Candidate variant: **KV-{v}**", "",
              f"## 2. Decision (docs/phase1_criteria.md): KV-{v} vs kmeans at equal k", "",
              f"**Band: {dec['band']}**", "", "| k | design | noise AUC diff | clean diff | band at k |",
              "|---|---|---|---|---|"]
    for k, x in dec["per_budget"].items():
        lines.append(f"| {k} | {x['design']} | {na._diff(x['noise'])} | {na._diff(x['clean'])} | {x['band']} |")

    lines += ["", "## 3. Descriptive comparisons (noise AUC unless stated; KV variants at their tune-selected gamma, "
              "Stage 3 designs at the candidate's gamma)", "",
              "| Comparison | " + " | ".join(f"k = {k}" for k in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    rows = [
        ("KV-S - na_fl_B, same keys (mechanism: what is returned)",
         lambda k: na.paired(res, chosen(raw, "S", k), na.na_name("B", sel["gamma"]["S"][str(k)], k), "noise_auc")),
        ("KV-S - na_fl_B, same keys, clean",
         lambda k: na.paired(res, chosen(raw, "S", k), na.na_name("B", sel["gamma"]["S"][str(k)], k), "clean")),
        ("KV-M - KV-S", lambda k: na.paired(res, chosen(raw, "M", k), chosen(raw, "S", k), "noise_auc")),
        (f"KV-{v} - na_fl_A", lambda k: na.paired(res, chosen(raw, v, k), at_candidate_gamma(raw, "A", k),
                                                  "noise_auc")),
        (f"KV-{v} - na_kmeans", lambda k: na.paired(res, chosen(raw, v, k), f"na_kmeans_k{k}", "noise_auc")),
        (f"KV-{v} - facility_location",
         lambda k: na.paired(res, chosen(raw, v, k), f"facility_location_k{k}", "noise_auc")),
        ("na_fl_B - kmeans", lambda k: na.paired(res, at_candidate_gamma(raw, "B", k), f"kmeans_k{k}", "noise_auc")),
        ("na_fl_A - kmeans", lambda k: na.paired(res, at_candidate_gamma(raw, "A", k), f"kmeans_k{k}", "noise_auc")),
    ]
    for t in na.KNOWN_TYPES + (na.TRANSFER_TYPE,):
        rows.append((f"KV-{v} - kmeans, {t} AUC",
                     lambda k, t=t: na.paired(res, chosen(raw, v, k), f"kmeans_k{k}", f"{t}_auc")))
    rows.append((f"KV-{v} at k - kmeans at 2k (equal stored vectors)",
                 lambda k: na.paired(res, chosen(raw, v, k), f"kmeans_k{2 * k}", "noise_auc")
                 if 2 * k in BUDGETS else None))
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join("—" if (d := fn(k)) is None else na._diff(d) for k in BUDGETS)
                     + " |")

    lines += ["", "## 4. Fraction of keys that are corrupted embeddings (mean over seeds)", "",
              "| gamma | " + " | ".join(f"k = {k}" for k in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for g in GAMMAS:
        cells = []
        for k in BUDGETS:
            vals = [d["noisy_keys"] for d in info.get(kv_name("S", g, k), [])]
            cells.append(f"{mean_std(vals)[0]:.3f}" if vals else "—")
        lines.append(f"| {g:g} | " + " | ".join(cells) + " |")

    a, b = chosen(raw, v, PER_LEVEL_K), f"{REFERENCE}_k{PER_LEVEL_K}"
    lines += ["", f"## 5. Per-level accuracy, {a} - {b} (pp; level 0 = clean)", "",
              "| Noise type | level | diff |", "|---|---|---|"]
    for kind, levels in raw["grids"].items():
        for i, level in enumerate(levels):
            if level != 0:
                lines.append(f"| {kind} | {level:g} | {na._diff(level_diff(raw, a, b, kind, i))} |")

    gaps_noise, gaps_clean = ms.memory_gaps(raw, "noise_auc"), ms.memory_gaps(raw, "clean")
    lines += ["", "## 6. Training / validation record (descriptive; training queries: 5000 per seed, noise stream 3, "
              "beta from validation)", "",
              "| Design | train clean | val clean | train noise AUC | val noise AUC | memory gap, noise AUC (pp) | "
              "memory gap, clean (pp) | queries stored as keys |", "|---|---|---|---|---|---|---|---|"]
    shown = ["no_memory"] + [f"{base}_k{k}" for base in ("kmeans", "facility_location", "na_kmeans") for k in BUDGETS]
    shown += [chosen(raw, vv, k) for vv in KV_VARIANTS for k in BUDGETS]
    shown += [at_candidate_gamma(raw, vv, k) for vv in ("A", "B") for k in BUDGETS]
    for name in dict.fromkeys(shown):
        tr, va = raw["train_results"][name], res[name]
        stored = raw["train_stored_fraction"].get(name)
        lines.append(
            f"| {name} | {mean_std([r['clean'] for r in tr])[0]:.4f} | {mean_std([r['clean'] for r in va])[0]:.4f} | "
            f"{mean_std([r['noise_auc'] for r in tr])[0]:.4f} | {mean_std([r['noise_auc'] for r in va])[0]:.4f} | "
            f"{'—' if name == 'no_memory' else na._diff(gaps_noise[name])} | "
            f"{'—' if name == 'no_memory' else na._diff(gaps_clean[name])} | "
            f"{'—' if stored is None else f'{mean_std(stored)[0]:.4f}'} |")
    lines += ["", "## 7. Absolute validation values (mean ± std over seeds)", "",
              "| Design | clean | noise AUC | known-type AUC | occlusion AUC | beta* |", "|---|---|---|---|---|---|"]
    for name in design_names():
        rr = res[name]
        beta = raw["beta_star"].get(name)
        c, n = mean_std([r["clean"] for r in rr]), mean_std([r["noise_auc"] for r in rr])
        kn, oc = mean_std([na.known_auc(r) for r in rr]), mean_std([r[f"{na.TRANSFER_TYPE}_auc"] for r in rr])
        lines.append(f"| {name} | {c[0]:.4f} ± {c[1]:.4f} | {n[0]:.4f} ± {n[1]:.4f} | {kn[0]:.4f} | {oc[0]:.4f} | "
                     f"{'—' if beta is None else f'{beta:g}'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("Stage 4: key-value noise-aware facility location screening (validation only)."
                            ).parse_args(argv)
    cfg = config_from_args(args)
    raw = run_key_value(cfg, retrain=args.retrain)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "key_value_screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
