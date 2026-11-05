"""RQ2 benchmark: bounded memory maintenance on a labelled embedding stream.

Protocol (frozen before any RQ2 test evaluation, gate G6):

* Scenario: initial memory = k-means (8 per class) on the first n0 training
  embeddings per class of a seeded order; the next 1,000 per class arrive,
  interleaved, in that order. n0 in {10, 30} primary, {100, 300} descriptive.
  3 stream orders per seed x 10 seeds; metrics are averaged over orders within
  a seed, then paired by seed (n = 10).
* Retrieval centre: the initial pool's mean, fixed, shared by every policy.
* beta: tuned per (policy, n0) on full validation with the repository rule
  (seed-and-order-mean validation head_acc over beta_selection_sets).
* Primary (test split): ra_fl vs frozen, n0 in {10, 30} x {Gaussian AUC,
  salt-and-pepper AUC}, Holm over these 4. Win = mean > 0, more positive seeds,
  Holm-adjusted exact Wilcoxon p < 0.05, clean accuracy >= frozen - 0.005.
* Secondary: ra_fl vs fifo, reservoir, fl_swap, ra (Holm within each
  comparator's 4 tests); recovery = (ra_fl - frozen) / (ceiling - frozen) with
  ceiling = k-means on everything seen; stream-order sensitivity; costs.
* Validation only, descriptive: robust accuracy at 10/25/50/100 % of the stream.

    python -m benchmarks.memory_maintenance_benchmark --reports-dir <REPORTS_DIR>
    python -m benchmarks.memory_maintenance_benchmark --quick
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.memory_diagnostics import paired_summary
from benchmarks.memory_maintenance import POLICY_CLASSES, stream_split
from benchmarks.memory_selection_benchmark import evaluate_test, holm_adjust
from benchmarks.mnist_baseline import (
    build_arg_parser,
    config_from_args,
    forward_embeddings,
    load_mnist,
    mean_std,
    noise_grids,
    prepare_baseline_run,
    resolve_device,
    tune_baseline,
)
from benchmarks.mnist_prototype_retrieval import beta_curve, build_hopfield_net, build_memory, select_beta

PER_CLASS = 8
STREAM_PER_CLASS = 1000
START_SIZES = (10, 30, 100, 300)
PRIMARY_STARTS = (10, 30)
ORDERS = (0, 1, 2)
POLICIES = ("frozen", "fifo", "reservoir", "fl_swap", "ra", "ra_fl")
CEILING = "kmeans_seen"
PROPOSED = "ra_fl"
COMPARATORS = ("fifo", "reservoir", "fl_swap", "ra")
PRIMARY_METRICS = ("gaussian_auc", "salt_pepper_auc")
CHECKPOINTS = (0.1, 0.25, 0.5, 1.0)
CHECK_SETS = (("gaussian", 0.3), ("gaussian", 0.5), ("salt_pepper", 0.1), ("salt_pepper", 0.2), ("occlusion", 10))
REF_SIZE = 200
LAYOUT_EVERY = 50
EPSILON = 0.0
NON_INFERIORITY_MARGIN = 0.005
ALPHA = 0.05
ORDER_SALT = 7_340_033
LOG_RUN = (0, 0, 10)  # (seed, order, n0) whose full decision log is saved


def initial_memory(z: torch.Tensor, y: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k-means patterns [C, K, D] and counts [C, K] (k-means members per centroid, at least 1)."""
    memory, labels, _ = build_memory(z, y, per_class=PER_CLASS, center=True, seed=seed)
    C = int(labels.max()) + 1
    patterns = memory.reshape(C, PER_CLASS, -1)
    counts = torch.ones(C, PER_CLASS)
    for c in range(C):
        assign = torch.cdist(z[y == c].double(), patterns[c].double()).argmin(dim=1)
        counts[c] = torch.bincount(assign, minlength=PER_CLASS).double().clamp_min(1.0)
    return patterns, counts


def run_stream(policy: Any, z_stream: torch.Tensor, y_stream: torch.Tensor) -> tuple[list[torch.Tensor], float]:
    """Feed the stream; return memory snapshots at CHECKPOINTS and microseconds per arrival.

    Only ``observe`` is timed; the invariant check after every arrival is not.
    """
    marks = {max(1, round(f * z_stream.shape[0])): i for i, f in enumerate(CHECKPOINTS)}
    snaps: list[torch.Tensor] = [torch.empty(0)] * len(CHECKPOINTS)
    elapsed = 0.0
    for t in range(z_stream.shape[0]):
        start = time.perf_counter()
        policy.observe(z_stream[t], int(y_stream[t]))
        elapsed += time.perf_counter() - start
        policy.check_invariants()
        if t + 1 in marks:
            snaps[marks[t + 1]] = policy.memory()[0]
    return snaps, 1e6 * elapsed / max(1, z_stream.shape[0])


def stream_plan(smallest: int) -> tuple[int, list[int]]:
    """(stream per class, feasible start sizes). Full data: 1,000 per class and every start.

    When classes are too small (quick mode), the stream shrinks to half the
    smallest class and only starts that still fit are run (reported).
    """
    starts = [n0 for n0 in START_SIZES if n0 + STREAM_PER_CLASS <= smallest]
    if starts == list(START_SIZES):
        return STREAM_PER_CLASS, starts
    per_class = smallest // 2
    return per_class, [n0 for n0 in START_SIZES if n0 + per_class <= smallest and n0 > PER_CLASS]


def _get(d: dict[Any, Any], k: Any) -> Any:
    """Lookup tolerant of int keys turned into strings by a JSON round trip."""
    return d[k] if k in d else d[str(k)]


def val_robust(net: Any, run: Any) -> float:
    labels = run.splits.val_labels
    accs = []
    for kind, level in CHECK_SETS:
        out = forward_embeddings(net, run.val_embeddings(kind, level), batch_size=run.batch_size, device=run.device)
        accs.append((out["head_pred"] == labels).float().mean().item())
    return sum(accs) / len(accs)


def run_benchmark(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = load_mnist(cfg["data_dir"])
    tune_baseline(cfg, data, device, retrain=retrain)
    betas, sel_sets, grids = cfg["beta_values"], cfg["beta_selection_sets"], noise_grids(cfg)
    designs = POLICIES + (CEILING,)

    # Stage 1: streams, final memories, checkpoints, validation beta curves.
    runs, cells = [], []   # cells: one per (seed, order, n0)
    curves: dict[tuple[str, int], list[list[float]]] = {}
    stats: dict[tuple[str, int], list[dict[str, Any]]] = {}
    decision_log: dict[str, list[dict[str, Any]]] = {}
    stream_sizes: set[int] = set()
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        runs.append(run)
        z, y = run.z_train, run.splits.train_labels
        per_class, starts = stream_plan(int(torch.bincount(y).min()))
        stream_sizes.add(per_class)
        if starts != list(START_SIZES):
            print(f"  seed {seed}: reduced plan, stream {per_class} per class, starts {starts}")
        for order in ORDERS:
            for n0 in starts:
                init, stream = stream_split(y, n0, per_class, seed * 1000 + order + ORDER_SALT)
                mu = z[init].mean(dim=0)
                patterns, counts = initial_memory(z[init], y[init], seed)
                cell = {"seed": seed, "order": order, "n0": n0, "mu": mu, "memories": {}, "snapshots": {}}
                for name in POLICIES:
                    log = (seed, order, n0) == LOG_RUN and name in ("ra_fl", "fl_swap")
                    policy = POLICY_CLASSES[name](patterns, counts, mu, z[init], y[init], seed=seed * 7 + order,
                                                  ref_size=REF_SIZE, layout_every=LAYOUT_EVERY,
                                                  epsilon=EPSILON, log=log)
                    snaps, us = run_stream(policy, z[stream], y[stream])
                    cell["memories"][name] = policy.memory()
                    cell["snapshots"][name] = snaps
                    stats.setdefault((name, n0), []).append(
                        {**policy.stats, "us_per_arrival": us, "extra_state_floats": policy.extra_state_floats()})
                    if policy.log is not None:
                        decision_log[name] = policy.log
                seen = torch.cat([init, stream])
                memory, labels, _ = build_memory(z[seen], y[seen], per_class=PER_CLASS, center=True, seed=seed)
                cell["memories"][CEILING] = (memory, labels)
                for name in designs:
                    memory, labels = cell["memories"][name]
                    net = build_hopfield_net(run.net, memory, labels, beta=betas[0], center=mu)
                    curves.setdefault((name, n0), []).append(beta_curve(net, run, betas, sel_sets))
                cells.append(cell)
                print(f"  seed {seed} order {order} n0={n0}: streamed {stream.numel()} embeddings")

    beta_star = {f"{n}|{n0}": select_beta(betas, c)[0] for (n, n0), c in curves.items()}
    print("beta*: " + ", ".join(f"{k}={v:g}" for k, v in beta_star.items()))

    # Stage 2: test evaluation of final memories; validation checkpoints.
    test: dict[str, dict[int, dict[int, list[dict[str, Any]]]]] = {}
    checkpoints: dict[str, dict[int, list[list[float]]]] = {}
    by_seed = {r.seed: r for r in runs}
    for cell in cells:
        run, n0 = by_seed[cell["seed"]], cell["n0"]
        for name in designs:
            memory, labels = cell["memories"][name]
            net = build_hopfield_net(run.net, memory, labels, beta=beta_star[f"{name}|{n0}"], center=cell["mu"])
            res = evaluate_test(net, run, grids)
            test.setdefault(name, {}).setdefault(n0, {}).setdefault(cell["seed"], []).append(res)
            if name in POLICIES:
                row = []
                for snap in cell["snapshots"][name]:
                    snet = build_hopfield_net(run.net, snap, labels, beta=beta_star[f"{name}|{n0}"], center=cell["mu"])
                    row.append(val_robust(snet, run))
                checkpoints.setdefault(name, {}).setdefault(n0, []).append(row)
        print(f"  seed {cell['seed']} order {cell['order']} n0={n0}: evaluated")

    return {
        "seeds": list(cfg["seeds"]), "orders": list(ORDERS), "starts": sorted({c["n0"] for c in cells}),
        "stream_per_class": sorted(stream_sizes), "beta_grid": list(betas), "beta_star": beta_star,
        "test": test, "checkpoints": checkpoints,
        "stats": {f"{n}|{n0}": v for (n, n0), v in stats.items()},
        "decision_log": decision_log, "grids": grids,
        "protocol": {"primary": f"{PROPOSED} vs frozen, n0 {list(PRIMARY_STARTS)} x {list(PRIMARY_METRICS)}, Holm over 4",
                     "non_inferiority_margin": NON_INFERIORITY_MARGIN, "alpha": ALPHA,
                     "ref_size": REF_SIZE, "layout_every": LAYOUT_EVERY, "epsilon": EPSILON},
    }


def seed_means(raw: dict[str, Any], name: str, n0: int, metric: str) -> list[float]:
    """Per-seed value of a metric, averaged over stream orders (seeds in raw['seeds'] order)."""
    per_seed = _get(raw["test"][name], n0)
    out = []
    for s in raw["seeds"]:
        rows = _get(per_seed, s)
        out.append(sum(r[metric] for r in rows) / len(rows))
    return out


def comparisons(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for other in ("frozen",) + COMPARATORS:
        family = []
        for n0 in [n for n in PRIMARY_STARTS if n in raw["starts"]]:
            clean = [a - b for a, b in zip(seed_means(raw, PROPOSED, n0, "clean"), seed_means(raw, other, n0, "clean"))]
            for metric in PRIMARY_METRICS:
                d = [a - b for a, b in zip(seed_means(raw, PROPOSED, n0, metric), seed_means(raw, other, n0, metric))]
                family.append({"vs": other, "n0": n0, "metric": metric, "primary": other == "frozen",
                               "clean_delta": sum(clean) / len(clean), **paired_summary(d)})
        for row, adj in zip(family, holm_adjust([r["wilcoxon_p"] for r in family])):
            row["holm_p"] = adj
            sig = adj is not None and adj < ALPHA
            ni = row["clean_delta"] >= -NON_INFERIORITY_MARGIN
            if sig and row["mean"] > 0 and row["n_pos"] > row["n_neg"] and ni:
                row["verdict"] = "WIN"
            elif sig and row["mean"] < 0 and row["n_neg"] > row["n_pos"]:
                row["verdict"] = "LOSS"
            else:
                row["verdict"] = "no significant difference"
        rows.extend(family)
    return rows


def _ms(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _p(p: float | None) -> str:
    return "—" if p is None else f"{p:.4f}"


def build_report(raw: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    designs = POLICIES + (CEILING,)
    starts = list(raw["starts"])
    lines = [
        "# RQ2 memory maintenance benchmark",
        "",
        f"Seeds {raw['seeds']} × orders {raw['orders']}; stream {raw['stream_per_class']} per class after a start of "
        f"n0 per class; k-means start with {PER_CLASS} patterns per class. Metrics averaged over orders within a "
        "seed, then paired by seed. Test split for final memories; validation for β and checkpoints.",
        f"Primary: {raw['protocol']['primary']}; win = mean > 0, more positive seeds, Holm p < {ALPHA}, clean Δ ≥ "
        f"−{NON_INFERIORITY_MARGIN} vs frozen.",
        "",
        "## 1. Test results per design (mean ± std over seeds)",
        "",
        "| n0 | Design | β* | Clean acc | Gaussian AUC | S&P AUC | Occlusion AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for n0 in starts:
        for name in designs:
            lines.append(f"| {n0} | {name} | {raw['beta_star'][f'{name}|{n0}']:g} | "
                         f"{_ms(seed_means(raw, name, n0, 'clean'))} | {_ms(seed_means(raw, name, n0, 'gaussian_auc'))} | "
                         f"{_ms(seed_means(raw, name, n0, 'salt_pepper_auc'))} | "
                         f"{_ms(seed_means(raw, name, n0, 'occlusion_auc'))} |")

    lines += ["", f"## 2. {PROPOSED} vs other policies (test; Holm within each comparator's 4 tests)", "",
              "| vs | n0 | Metric | Δ (pp) | 95 % CI (pp) | +/− | Wilcoxon p | Holm p | Δ clean (pp) | Verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = " (primary)" if r["primary"] else ""
        lines.append(f"| {r['vs']}{tag} | {r['n0']} | {r['metric']} | {100 * r['mean']:+.3f} | "
                     f"[{100 * r['ci_low']:+.3f}, {100 * r['ci_high']:+.3f}] | {r['n_pos']}/{r['n_neg']} | "
                     f"{_p(r['wilcoxon_p'])} | {_p(r['holm_p'])} | {100 * r['clean_delta']:+.3f} | {r['verdict']} |")

    lines += ["", f"## 3. Gap recovered (test): (policy − frozen) / ({CEILING} − frozen)", "",
              "| n0 | Metric | Gap frozen→ceiling (pp) | " + " | ".join(POLICIES[1:]) + " |",
              "|---|---|---|" + "---|" * (len(POLICIES) - 1)]
    for n0 in starts:
        for metric in PRIMARY_METRICS:
            fz = sum(seed_means(raw, "frozen", n0, metric)) / len(raw["seeds"])
            ce = sum(seed_means(raw, CEILING, n0, metric)) / len(raw["seeds"])
            gap = ce - fz
            vals = []
            for name in POLICIES[1:]:
                v = sum(seed_means(raw, name, n0, metric)) / len(raw["seeds"])
                vals.append(f"{(v - fz) / gap:+.2f}" if abs(gap) > 1e-9 else "—")
            lines.append(f"| {n0} | {metric} | {100 * gap:+.3f} | " + " | ".join(vals) + " |")

    lines += ["", "## 4. Validation robust accuracy along the stream (mean over seeds and orders)", "",
              "| n0 | Policy | " + " | ".join(f"{int(100 * f)} %" for f in CHECKPOINTS) + " |",
              "|---|---|" + "---|" * len(CHECKPOINTS)]
    for n0 in starts:
        for name in POLICIES:
            rows_c = _get(raw["checkpoints"][name], n0)
            means = [sum(r[i] for r in rows_c) / len(rows_c) for i in range(len(CHECKPOINTS))]
            lines.append(f"| {n0} | {name} | " + " | ".join(f"{m:.4f}" for m in means) + " |")

    lines += ["", "## 5. Stream-order sensitivity (test Gaussian AUC; mean over seeds of the SD across orders, pp)", "",
              "| n0 | " + " | ".join(designs) + " |", "|---|" + "---|" * len(designs)]
    for n0 in starts:
        vals = []
        for name in designs:
            per_seed = _get(raw["test"], name)
            per_seed = _get(per_seed, n0)
            sds = [mean_std([r["gaussian_auc"] for r in _get(per_seed, s)])[1] for s in raw["seeds"]]
            vals.append(f"{100 * sum(sds) / len(sds):.3f}")
        lines.append(f"| {n0} | " + " | ".join(vals) + " |")

    lines += ["", "## 6. Decisions and cost per run (mean over seeds and orders)", "",
              "| n0 | Policy | Changed / arrivals | Swaps | Absorbed | Layout changes / checks | Conflicts | µs per arrival | Extra state (floats) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n0 in starts:
        for name in POLICIES:
            st = raw["stats"][f"{name}|{n0}"]

            def m(k: str) -> float:
                return sum(s[k] for s in st) / len(st)

            lines.append(f"| {n0} | {name} | {m('changed'):.0f} / {m('arrivals'):.0f} | {m('swaps'):.0f} | {m('absorbed'):.0f} | "
                         f"{m('layout_changes'):.1f} / {m('layout_checks'):.0f} | {m('conflicts'):.0f} | "
                         f"{m('us_per_arrival'):.0f} | {m('extra_state_floats'):.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("RQ2: memory maintenance benchmark.").parse_args(argv)
    cfg = config_from_args(args)
    raw = run_benchmark(cfg, retrain=args.retrain)
    rows = comparisons(raw)
    report = build_report(raw, rows)
    out_dir = Path(cfg["reports_dir"]) / "memory_maintenance"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    log = raw.pop("decision_log")
    with open(out_dir / "decision_log.jsonl", "w", encoding="utf-8") as f:
        for name, entries in log.items():
            for e in entries:
                f.write(json.dumps({"policy": name, **e}, default=float) + "\n")
    raw["comparisons"] = rows
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
