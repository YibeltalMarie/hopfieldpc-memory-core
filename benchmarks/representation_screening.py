"""Stage 2: selection strategy x representation (real example vs region mean), validation only.

Each selection rule chooses k regions per class; the representation decides
what is stored for each region (proposal Section 5.9, Eq. 5.14):

    real          the selected training embedding itself
    region mean   the mean of the class's training embeddings assigned to it
                  (one assign-and-average step, centred-cosine assignment)

Designs at every budget k in BUDGETS (all class-wise, same centre and retrieval):

    kmeans_k{k}                    k-means centroids (repository default; means)
    kmeans_medoid_k{k}             real embedding closest to each centroid (own cluster)
    facility_location_k{k}         greedy FL picks (real)
    facility_location_rm_k{k}      region means of the FL picks           <- candidate method
    facility_location_rm_euc_k{k}  the same with Euclidean assignment     (ablation A8)
    random_k{k}                    random picks (real; control)
    random_rm_k{k}                 region means of the random picks (control)
    no_memory                      the classifier head on z

Protocol: memory_screening's (beta on the validation-tune half, metrics on the
validation-eval half, paired by seed; test split never read). Only
facility_location_rm is judged by the Phase 1 bands, against kmeans at the same
k; every other comparison is descriptive.

    python -m benchmarks.representation_screening --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks import memory_screening as ms
from benchmarks.memory_diagnostics import medoid_representatives, paired_summary
from benchmarks.memory_diagnostics import region_means as cosine_region_means
from benchmarks.memory_selection import nested_prefix, select_class_memory
from benchmarks.mnist_baseline import mean_std
from benchmarks.mnist_prototype_retrieval import build_memory
from benchmarks.screening_criteria import method_band, band_at_budget
from benchmarks.runs import build_arg_parser, config_from_args

BUDGETS = ms.BUDGETS
BASES = ("kmeans", "kmeans_medoid", "facility_location", "facility_location_rm", "facility_location_rm_euc",
         "random", "random_rm")
CANDIDATE, REFERENCE = "facility_location_rm", "kmeans"
PAIRS = (
    ("facility_location_rm", "facility_location", "Averaging, FL regions (mean - real)"),
    ("random_rm", "random", "Averaging, random regions (mean - real)"),
    ("kmeans", "kmeans_medoid", "Averaging, k-means regions (centroid - medoid)"),
    ("facility_location_rm", "kmeans", "FL regions vs k-means regions (both means)"),
    ("facility_location_rm_euc", "facility_location_rm", "Euclidean vs cosine assignment (ablation A8)"),
)


def design_names() -> list[str]:
    return ["no_memory"] + [f"{base}_k{k}" for base in BASES for k in BUDGETS]


def euclidean_region_means(z: torch.Tensor, labels: torch.Tensor, picks: torch.Tensor,
                           pick_labels: torch.Tensor) -> torch.Tensor:
    """Region means with Euclidean (raw-embedding) assignment; an empty region keeps its pick."""
    means = picks.clone()
    for c in pick_labels.unique(sorted=True).tolist():
        idx = torch.nonzero(pick_labels == c, as_tuple=True)[0]
        zc = z[labels == c]
        assign = torch.cdist(zc.double(), picks[idx].double()).argmin(dim=1)
        for local, j in enumerate(idx.tolist()):
            members = zc[assign == local]
            if members.shape[0] > 0:
                means[j] = members.mean(dim=0)
    return means


def build_designs(run: Any, device: torch.device) -> dict[str, dict[str, Any]]:
    """Every Stage 2 memory: patterns, labels, centre and construction seconds."""
    z, y = run.z_train, run.splits.train_labels
    center = z.mean(dim=0)
    designs: dict[str, dict[str, Any]] = {}

    def add(name: str, memory: torch.Tensor, labels: torch.Tensor, secs: float) -> None:
        designs[name] = {"memory": memory.cpu(), "labels": labels.cpu(), "center": center, "build_s": secs}

    for k in BUDGETS:
        start = time.perf_counter()
        centroids, cent_labels, _ = build_memory(z, y, per_class=k, center=True, seed=run.seed)
        secs = time.perf_counter() - start
        add(f"kmeans_k{k}", centroids, cent_labels, secs)
        start = time.perf_counter()
        medoids, _ = medoid_representatives(z, y, centroids, cent_labels, center, own_cluster=True)
        add(f"kmeans_medoid_k{k}", medoids, cent_labels, secs + time.perf_counter() - start)
    k_max = max(BUDGETS)
    for method in ("facility_location", "random"):   # nested: one run at k_max gives every budget
        dev = device if method == "facility_location" else torch.device("cpu")
        start = time.perf_counter()
        sel, sel_labels, _ = select_class_memory(z.to(dev), y.to(dev), per_class=k_max, method=method,
                                                 center=center.to(dev), seed=run.seed)
        secs = time.perf_counter() - start
        sel, sel_labels = sel.cpu(), sel_labels.cpu()
        for k in BUDGETS:
            picks, pick_labels = nested_prefix(sel, sel_labels, per_class_max=k_max, per_class=k)
            add(f"{method}_k{k}", picks, pick_labels, secs)
            start = time.perf_counter()
            means, _ = cosine_region_means(z, y, picks, pick_labels, center)
            add(f"{method}_rm_k{k}", means, pick_labels, secs + time.perf_counter() - start)
            if method == "facility_location":
                start = time.perf_counter()
                means = euclidean_region_means(z, y, picks, pick_labels)
                add(f"{method}_rm_euc_k{k}", means, pick_labels, secs + time.perf_counter() - start)
    return designs


def run_representation(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    return ms.run_screening(cfg, retrain=retrain, builder=build_designs, names=design_names())


def paired(raw: dict[str, Any], a: str, b: str, metric: str) -> dict[str, Any]:
    """Paired differences a - b over seeds."""
    return paired_summary([ra[metric] - rb[metric] for ra, rb in zip(raw["results"][a], raw["results"][b])])


def candidate_decision(raw: dict[str, Any]) -> dict[str, Any]:
    per_k = {}
    for k in BUDGETS:
        a, b = f"{CANDIDATE}_k{k}", f"{REFERENCE}_k{k}"
        noise, clean = paired(raw, a, b, "noise_auc"), paired(raw, a, b, "clean")
        per_k[str(k)] = {"noise": noise, "clean": clean, "band": band_at_budget(noise, clean)}
    return {"per_budget": per_k, "band": method_band({k: v["band"] for k, v in per_k.items()})}


def _diff(s: dict[str, Any]) -> str:
    return (f"{100 * s['mean']:+.2f} [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}] "
            f"({s['n_pos']}/{s['n_neg']})")


def build_report(raw: dict[str, Any]) -> str:
    dec = candidate_decision(raw)
    lines = [f"# Stage 2: selection x representation — {raw['dataset']} (validation only)", "",
             f"Seeds {raw['seeds']}; budgets k = {list(BUDGETS)} per class; beta per design on the tuning half; "
             "metrics on the evaluation half; differences in pp, mean [95% t-CI] (seeds +/-). "
             "The test split was not read.", "",
             f"## 1. Decision (docs/phase1_criteria.md): {CANDIDATE} vs {REFERENCE} at equal k", "",
             f"**Band: {dec['band']}**", "",
             "| k | noise AUC diff | clean diff | band at k |", "|---|---|---|---|"]
    for k, v in dec["per_budget"].items():
        lines.append(f"| {k} | {_diff(v['noise'])} | {_diff(v['clean'])} | {v['band']} |")
    lines += ["", "## 2. Representation effects (descriptive), noise AUC", "",
              "| Comparison | " + " | ".join(f"k = {k}" for k in BUDGETS) + " |",
              "|---|" + "---|" * len(BUDGETS)]
    for a, b, label in PAIRS:
        cells = [_diff(paired(raw, f"{a}_k{k}", f"{b}_k{k}", "noise_auc")) for k in BUDGETS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## 3. Absolute values (mean ± std over seeds)", "",
              "| Design | clean | noise AUC | beta* |", "|---|---|---|---|"]
    for name in design_names():
        rows = raw["results"][name]
        clean, noise = mean_std([r["clean"] for r in rows]), mean_std([r["noise_auc"] for r in rows])
        beta = raw["beta_star"].get(name)
        lines.append(f"| {name} | {clean[0]:.4f} ± {clean[1]:.4f} | {noise[0]:.4f} ± {noise[1]:.4f} | "
                     f"{'—' if beta is None else f'{beta:g}'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("Stage 2: selection x representation screening (validation only).").parse_args(argv)
    cfg = config_from_args(args)
    raw = run_representation(cfg, retrain=args.retrain)
    raw["decision"] = candidate_decision(raw)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "representation_screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
