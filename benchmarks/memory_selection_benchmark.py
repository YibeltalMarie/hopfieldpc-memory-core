"""RQ1 benchmark: which embeddings should the Hopfield memory store?

Compares, on the same frozen backbones (cached ``backbone_seed{s}.pt``):

* ``class_means``        - one mean per class (10 patterns; compact reference)
* ``kmeans``             - class-wise k-means centroids (existing method; primary target)
* ``facility_location``  - class-wise greedy facility location on real embeddings (C1)
* ``random``             - class-wise random real embeddings (control)

at 2, 4 and 8 patterns per class. Every design gets its own beta, selected on
validation data only with the repository protocol (seed-mean validation
head_acc over ``beta_selection_sets``; ties -> smallest beta). The test split
is used once, for the final numbers. All designs of a seed are evaluated on
identical test embeddings, so every comparison is paired per seed.

Protocol frozen before any test evaluation (gate G4):

* Primary: Gaussian and salt-and-pepper ``robust_auc`` of head accuracy,
  method minus k-means at the same budget, per noise type.
* Win at a budget and noise type: mean delta > 0, more positive than
  negative seeds, Holm-adjusted exact Wilcoxon p < 0.05, and clean accuracy
  non-inferior (mean delta >= -0.005). Holm family for the primary claim:
  one method x three budgets x two primary metrics (six tests). Secondary
  metrics are Holm-adjusted over the three budgets and are descriptive only.
* Pairing: every difference is d_i = metric(method, seed i) - metric(k-means,
  seed i) on the same backbone and identical test embeddings.
* beta tuning is identical for every design: same beta grid, same validation
  sets and noise draws, same metric (mean validation head_acc), every grid
  value evaluated (no early stopping), same rule (argmax of the seed-mean
  curve, ties -> smallest beta, one beta per design shared by all seeds).
  The validation curves are saved for audit.
* Secondary: occlusion AUC, clean accuracy, head agreement on stored
  patterns, validation coverage, construction time.

Pattern types: class means and k-means store synthetic averages; facility
location and random store real training embeddings. Facility location and
random are nested across budgets (S_2 within S_4 within S_8): one greedy run
at k=8 gives every budget, so the budgets are not independent experiments.
A separate greedy run at k=2 or k=4 returns exactly the same prefix (tested).
K-means is recomputed at each budget and is not nested.

Run from the repository root, with the reports directory that holds the
cached backbones:

    python -m benchmarks.memory_selection_benchmark --reports-dir <REPORTS_DIR>
    python -m benchmarks.memory_selection_benchmark --quick   # smoke run
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.memory_selection import class_coverage, nested_prefix, select_class_memory
from benchmarks.mnist_baseline import (
    NOISE_TYPES,
    build_arg_parser,
    config_from_args,
    forward_embeddings,
    load_mnist,
    mean_std,
    noise_grids,
    prepare_baseline_run,
    resolve_device,
    robust_auc,
    tune_baseline,
)
from benchmarks.mnist_prototype_retrieval import (
    beta_curve,
    build_hopfield_net,
    build_memory,
    seed_stats,
    select_beta,
)

BUDGETS = (2, 4, 8)
METHODS = ("kmeans", "facility_location", "random")
CHALLENGERS = ("facility_location", "random")
PRIMARY = ("gaussian", "salt_pepper")
PRIMARY_METRICS = tuple(f"{k}_auc" for k in PRIMARY)
PATTERN_TYPE = {
    "class_means": "synthetic mean",
    "kmeans": "synthetic centroids",
    "facility_location": "real embeddings",
    "random": "real embeddings",
}
NON_INFERIORITY_MARGIN = 0.005
ALPHA = 0.05


def design_names() -> list[str]:
    return ["class_means"] + [f"{m}_k{k}" for k in BUDGETS for m in METHODS]


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values; None entries are left out of the family."""
    present = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[float | None] = [None] * len(p_values)
    m, running = len(present), 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def build_designs(run: Any, center: bool, device: torch.device) -> dict[str, dict[str, Any]]:
    """Every memory design of one seed: patterns, labels, centre, construction time."""
    z, y = run.z_train, run.splits.train_labels
    designs: dict[str, dict[str, Any]] = {}

    def add(name: str, memory: torch.Tensor, labels: torch.Tensor, offset: Any, secs: float) -> None:
        designs[name] = {"memory": memory.cpu(), "labels": labels.cpu(), "center": offset,
                         "build_s": secs}

    start = time.perf_counter()
    memory, labels, offset = build_memory(z, y, per_class=1, center=center, seed=run.seed)
    add("class_means", memory, labels, offset, time.perf_counter() - start)

    for k in BUDGETS:
        for method in ("kmeans", "random"):
            start = time.perf_counter()
            memory, labels, offset = build_memory(z, y, per_class=k, center=center,
                                                  seed=run.seed, method=method)
            add(f"{method}_k{k}", memory, labels, offset, time.perf_counter() - start)

    # Facility location once at the largest budget; smaller budgets are its prefixes.
    k_max = max(BUDGETS)
    offset = z.mean(dim=0) if center else None
    start = time.perf_counter()
    fl, fl_labels, _ = select_class_memory(
        z.to(device), y.to(device), per_class=k_max, method="facility_location",
        center=None if offset is None else offset.to(device), seed=run.seed,
    )
    fl_secs = time.perf_counter() - start
    for k in BUDGETS:
        memory, labels = nested_prefix(fl.cpu(), fl_labels.cpu(), per_class_max=k_max, per_class=k)
        add(f"facility_location_k{k}", memory, labels, offset, fl_secs)
    return designs


def head_accuracy(net: Any, run: Any, z: torch.Tensor, labels: torch.Tensor) -> float:
    out = forward_embeddings(net, z, batch_size=run.batch_size, device=run.device)
    return (out["head_pred"] == labels).float().mean().item()


def evaluate_test(net: Any, run: Any, grids: dict[str, list[float]]) -> dict[str, Any]:
    """Clean accuracy, per-level accuracy and robust_auc per noise type on the test split."""
    labels = run.splits.test_labels
    result: dict[str, Any] = {"clean": head_accuracy(net, run, run.test_embeddings("gaussian", 0.0), labels)}
    for kind in NOISE_TYPES:
        accs = [head_accuracy(net, run, run.test_embeddings(kind, level), labels) for level in grids[kind]]
        result[f"{kind}_acc"] = accs
        result[f"{kind}_auc"] = robust_auc(grids[kind], accs)
    return result


def run_benchmark(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = load_mnist(cfg["data_dir"])
    tune_baseline(cfg, data, device, retrain=retrain)  # cached; restores tuned lr / batch / epochs
    betas, sel_sets, grids = cfg["beta_values"], cfg["beta_selection_sets"], noise_grids(cfg)
    names = design_names()

    # Stage 1 (training + validation only): memories, diagnostics, beta curves.
    runs, nets_per_seed = [], []
    curves: dict[str, list[list[float]]] = {n: [] for n in names}
    diagnostics: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        print(f"  seed {seed}: building memories")
        designs = build_designs(run, cfg["center_prototypes"], device)
        nets = {}
        for name in names:
            d = designs[name]
            net = build_hopfield_net(run.net, d["memory"], d["labels"], beta=betas[0], center=d["center"])
            with torch.inference_mode():
                head_pred = run.net.head(d["memory"].to(run.device)).argmax(dim=-1).cpu()
            coverage = class_coverage(run.z_val, run.splits.val_labels, d["memory"], d["labels"], d["center"])
            diagnostics[name].append({
                "build_s": d["build_s"],
                "head_correct": int((head_pred == d["labels"]).sum().item()),
                "num_patterns": int(d["labels"].numel()),
                "val_coverage_mean": coverage.mean().item(),
                "val_coverage_min": coverage.min().item(),
            })
            curves[name].append(beta_curve(net, run, betas, sel_sets))
            nets[name] = net
        runs.append(run)
        nets_per_seed.append(nets)

    beta_star = {name: select_beta(betas, curves[name])[0] for name in names}
    print("beta* per design (validation only): "
          + ", ".join(f"{n}={b:g}" for n, b in beta_star.items()))

    # Stage 2: one paired test evaluation per seed and design.
    test: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    baseline: list[dict[str, Any]] = []
    for run, nets in zip(runs, nets_per_seed):
        print(f"  seed {run.seed}: test evaluation")
        baseline.append(evaluate_test(run.net, run, grids))
        for name in names:
            nets[name].bridge.set_retrieval(beta=beta_star[name], num_updates=1)
            test[name].append(evaluate_test(nets[name], run, grids))

    return {
        "config": cfg, "seeds": list(cfg["seeds"]), "budgets": list(BUDGETS), "grids": grids,
        "beta_grid": list(betas), "beta_selection_sets": sel_sets, "val_curves": curves,
        "beta_star": beta_star, "diagnostics": diagnostics, "test": test, "baseline": baseline,
        "protocol": {"primary": list(PRIMARY), "non_inferiority_margin": NON_INFERIORITY_MARGIN,
                     "alpha": ALPHA, "correction": "holm over six primary tests per method (3 budgets x 2 noise types); "
                     "secondary metrics holm over budgets, descriptive"},
    }


def compare(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Paired per-seed comparisons of each challenger with k-means at equal budget."""
    rows = []
    metrics = [f"{k}_auc" for k in NOISE_TYPES] + ["clean"]
    for method in CHALLENGERS:
        by_metric: dict[str, list[dict[str, Any]]] = {}
        for metric in metrics:
            for k in BUDGETS:
                ours, ref = raw["test"][f"{method}_k{k}"], raw["test"][f"kmeans_k{k}"]
                if len(ours) != len(ref):
                    raise ValueError(f"unpaired seeds for {method}_k{k} vs kmeans_k{k}.")
                deltas = [a[metric] - b[metric] for a, b in zip(ours, ref)]  # same seed i
                by_metric.setdefault(metric, []).append({
                    "method": method, "budget": k, "metric": metric,
                    "primary": metric in PRIMARY_METRICS,
                    "delta": sum(deltas) / len(deltas), "delta_sd": mean_std(deltas)[1],
                    **seed_stats(deltas),
                })
        families = [[r for m in PRIMARY_METRICS for r in by_metric[m]]]
        families += [by_metric[m] for m in metrics if m not in PRIMARY_METRICS]
        for family in families:
            for row, adj in zip(family, holm_adjust([r["wilcoxon_p"] for r in family])):
                row["holm_p"] = adj
        rows.extend(r for m in metrics for r in by_metric[m])
    clean = {(r["method"], r["budget"]): r["delta"] for r in rows if r["metric"] == "clean"}
    for row in rows:
        row["non_inferior"] = clean[(row["method"], row["budget"])] >= -NON_INFERIORITY_MARGIN
        row["verdict"] = verdict(row)
    return rows


def verdict(row: dict[str, Any]) -> str:
    if row["metric"] == "clean":
        return "non-inferior" if row["non_inferior"] else "inferior"
    p = row["holm_p"]
    significant = p is not None and p < ALPHA
    if not row["primary"]:
        if significant:
            return "higher (secondary)" if row["delta"] > 0 else "lower (secondary)"
        return "no significant difference"
    if significant and row["delta"] > 0 and row["n_pos"] > row["n_neg"] and row["non_inferior"]:
        return "WIN"
    if significant and row["delta"] < 0 and row["n_neg"] > row["n_pos"]:
        return "LOSS"
    return "no significant difference"


def _ms(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_report(raw: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    names = design_names()
    lines = [
        "# RQ1 memory selection benchmark",
        "",
        f"Seeds: {raw['seeds']}. Budgets per class: {raw['budgets']}. Beta selected per design on "
        "validation only; test split used once for these numbers.",
        f"Win rule (primary metrics only): mean Δ > 0, more positive seeds, Holm-adjusted Wilcoxon "
        f"p < {ALPHA}, clean accuracy Δ ≥ −{NON_INFERIORITY_MARGIN}. Holm family: one method × three "
        "budgets × two primary metrics (six tests); secondary metrics are Holm-adjusted over budgets "
        "and descriptive only.",
        "β tuning is identical for every design: same grid "
        f"{raw['beta_grid']}, same validation sets {raw['beta_selection_sets']}, "
        "every value evaluated, same rule (argmax of seed-mean validation head_acc, ties → smallest).",
        "Δ_i = metric(method, seed i) − metric(k-means, seed i) on the same backbone and test "
        "embeddings. With 10 seeds the power is limited: p-values are evidence, not proof.",
        "",
        "## Results per design (test, mean ± std over seeds)",
        "",
        "| Design | Pattern type | β* | Clean acc | Gaussian AUC | S&P AUC | Occlusion AUC | Head correct | Val coverage | Build s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    base = raw["baseline"]
    lines.append(f"| baseline (no memory) | — | — | {_ms([b['clean'] for b in base])} | "
                 f"{_ms([b['gaussian_auc'] for b in base])} | {_ms([b['salt_pepper_auc'] for b in base])} | "
                 f"{_ms([b['occlusion_auc'] for b in base])} | — | — | — |")
    for name in names:
        t, d = raw["test"][name], raw["diagnostics"][name]
        correct = sum(x["head_correct"] for x in d)
        total = sum(x["num_patterns"] for x in d)
        ptype = PATTERN_TYPE[name.rsplit("_k", 1)[0]]
        lines.append(
            f"| {name} | {ptype} | {raw['beta_star'][name]:g} | {_ms([x['clean'] for x in t])} | "
            f"{_ms([x['gaussian_auc'] for x in t])} | {_ms([x['salt_pepper_auc'] for x in t])} | "
            f"{_ms([x['occlusion_auc'] for x in t])} | {correct}/{total} | "
            f"{_ms([x['val_coverage_mean'] for x in d])} | {_ms([x['build_s'] for x in d], 2)} |"
        )
    lines += [
        "",
        "Facility location and random are nested across budgets (S_2 ⊂ S_4 ⊂ S_8, one selection path; "
        "a separate greedy run at a smaller budget gives the same prefix), so their budgets are not "
        "independent experiments. Facility-location build time is the single run at the largest budget. "
        "K-means is recomputed at every budget.",
        "",
        "## Paired comparisons with k-means at equal budget",
        "",
        "| Method | k/class | Metric | Δ mean (pp) | SD of Δ (pp) | Seeds +/− | Wilcoxon p | Holm p | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt_p(p: float | None) -> str:
        return "—" if p is None else f"{p:.4f}"

    for r in rows:
        lines.append(f"| {r['method']} | {r['budget']} | {r['metric']}{'' if r['primary'] else ' (secondary)'} | "
                     f"{100 * r['delta']:+.2f} | {100 * r['delta_sd']:.2f} | {r['n_pos']}/{r['n_neg']} | {fmt_p(r['wilcoxon_p'])} | {fmt_p(r['holm_p'])} | "
                     f"{r['verdict']} |")
    lines += ["", "Primary metrics are gaussian_auc and salt_pepper_auc; random is a control, not a "
              "proposed method."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("RQ1: memory selection benchmark (k-means vs facility location vs random).") \
        .parse_args(argv)
    cfg = config_from_args(args)
    raw = run_benchmark(cfg, retrain=args.retrain)
    rows = compare(raw)
    report = build_report(raw, rows)
    out_dir = Path(cfg["reports_dir"]) / "memory_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    serialisable = {k: v for k, v in raw.items() if k != "config"}
    serialisable["comparisons"] = rows
    (out_dir / "raw.json").write_text(json.dumps(serialisable, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
