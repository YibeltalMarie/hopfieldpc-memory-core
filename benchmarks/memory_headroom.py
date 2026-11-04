"""Headroom checks for RQ2 (validation only): how much could memory maintenance gain?

Check 1, small start: k-means (8 per class) built from only the first n0
training embeddings per class (fixed seeded stream order), frozen, versus
k-means on all training embeddings. The gap bounds what an update policy
fed the rest of the stream could add.

Check 2, adaptation: k-means built from clean training embeddings plus
embeddings of corrupted *training* images (noise stream 2; validation uses
stream 1 and test stream 0), versus clean-only k-means, measured on noisy
validation queries.

    matched     training noise at the evaluation kinds and levels
    mismatched  training noise at Gaussian 0.4 and salt-and-pepper 0.15 only (no occlusion)

The retrieval centre is the clean training mean for every design, beta is
tuned per design on the validation tuning half with the repository rule,
and every number is measured on the validation evaluation half.

    python -m benchmarks.memory_headroom --reports-dir <REPORTS_DIR>
    python -m benchmarks.memory_headroom --quick
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from benchmarks.memory_diagnostics import key, paired_summary, split_validation, tune_score
from benchmarks.mnist_baseline import (
    build_arg_parser,
    config_from_args,
    encode,
    forward_embeddings,
    load_mnist,
    make_corrupted_test_set,
    mean_std,
    prepare_baseline_run,
    resolve_device,
    tune_baseline,
)
from benchmarks.mnist_prototype_retrieval import build_hopfield_net, build_memory, select_beta

PER_CLASS = 8
START_SIZES = (10, 30, 100, 300)
TRAIN_NOISE_STREAM = 2
STREAM_SALT = 104729
MATCHED = (("gaussian", 0.3), ("gaussian", 0.5), ("salt_pepper", 0.1), ("salt_pepper", 0.2), ("occlusion", 10))
MISMATCHED = (("gaussian", 0.4), ("salt_pepper", 0.15))
EVAL_SETS = (("clean", 0.0),) + MATCHED
ROBUST_SETS = MATCHED
EXTRA_BETAS = (200.0, 500.0, 1000.0, 10000.0)
REFERENCE = "kmeans_full"


def first_n_per_class(labels: torch.Tensor, n0: int, seed: int) -> torch.Tensor:
    """Row indices of the first n0 embeddings of each class in a fixed seeded stream order."""
    order = torch.randperm(labels.numel(), generator=torch.Generator().manual_seed(seed + STREAM_SALT))
    keep = []
    for c in labels.unique(sorted=True).tolist():
        rows = order[labels[order] == c]
        if rows.numel() < n0:
            raise ValueError(f"class {c} has {rows.numel()} embeddings, fewer than n0={n0}.")
        keep.append(rows[:n0])
    return torch.cat(keep).sort().values


def corrupted_train_embeddings(run: Any, sets: tuple, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean training embeddings plus one corrupted copy per (kind, level), with labels."""
    zs, ys = [run.z_train], [run.splits.train_labels]
    for kind, level in sets:
        images = make_corrupted_test_set(run.splits.train_images, seed=run.seed, kind=kind,
                                         level=level, stream=TRAIN_NOISE_STREAM)
        zs.append(encode(run.net.encoder, images, batch_size=batch_size, device=run.device))
        ys.append(run.splits.train_labels)
    return torch.cat(zs), torch.cat(ys)


def build_designs(run: Any, batch_size: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    z, y = run.z_train, run.splits.train_labels
    designs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    memory, labels, _ = build_memory(z, y, per_class=PER_CLASS, center=True, seed=run.seed)
    designs[REFERENCE] = (memory, labels)
    smallest = int(torch.bincount(y).min().item())
    for n0 in START_SIZES:
        if n0 > smallest:
            print(f"  seed {run.seed}: skipping n0={n0} (smallest class has {smallest} embeddings)")
            continue
        rows = first_n_per_class(y, n0, run.seed)
        memory, labels, _ = build_memory(z[rows], y[rows], per_class=PER_CLASS, center=True, seed=run.seed)
        designs[f"kmeans_first{n0}"] = (memory, labels)
    for name, sets in (("kmeans_noisy_matched", MATCHED), ("kmeans_noisy_mismatched", MISMATCHED)):
        zp, yp = corrupted_train_embeddings(run, sets, batch_size)
        memory, labels, _ = build_memory(zp, yp, per_class=PER_CLASS, center=True, seed=run.seed)
        designs[name] = (memory, labels)
    return designs


def evaluate(net: Any, run: Any, ev: torch.Tensor) -> dict[str, float]:
    y = run.splits.val_labels[ev]
    out = {}
    for kind, level in EVAL_SETS:
        pred = forward_embeddings(net, run.val_embeddings(kind, level)[ev],
                                  batch_size=run.batch_size, device=run.device)["head_pred"]
        out[key(kind, level)] = (pred == y).float().mean().item()
    out["robust"] = sum(out[key(k, l)] for k, l in ROBUST_SETS) / len(ROBUST_SETS)
    return out


def run_headroom(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = load_mnist(cfg["data_dir"])
    tune_baseline(cfg, data, device, retrain=retrain)
    betas = list(cfg["beta_values"]) + [b for b in EXTRA_BETAS if b not in cfg["beta_values"]]
    sel_sets = cfg["beta_selection_sets"]

    state, curves = [], {}
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        tune, ev = split_validation(run.z_val.shape[0], seed)
        print(f"  seed {seed}: building designs (incl. corrupted training embeddings) and tuning beta")
        mu = run.z_train.mean(dim=0)
        nets = {}
        for name, (memory, labels) in build_designs(run, cfg["eval_batch_size"]).items():
            net = build_hopfield_net(run.net, memory, labels, beta=betas[0], center=mu)
            curve = []
            for beta in betas:
                net.bridge.set_retrieval(beta=beta, num_updates=1)
                curve.append(tune_score(net, run, sel_sets, tune))
            curves.setdefault(name, []).append(curve)
            nets[name] = net
        state.append((run, ev, nets))

    beta_star = {name: select_beta(betas, c)[0] for name, c in curves.items()}
    print("beta* (validation tuning half): " + ", ".join(f"{n}={b:g}" for n, b in beta_star.items()))

    results: dict[str, list[dict[str, float]]] = {}
    baseline: list[dict[str, float]] = []
    for run, ev, nets in state:
        print(f"  seed {run.seed}: measuring on {ev.numel()} validation images")
        baseline.append(evaluate(run.net, run, ev))
        for name, net in nets.items():
            net.bridge.set_retrieval(beta=beta_star[name], num_updates=1)
            results.setdefault(name, []).append(evaluate(net, run, ev))

    return {"seeds": list(cfg["seeds"]), "beta_grid": betas, "beta_star": beta_star,
            "val_curves_tune": curves, "results": results, "baseline": baseline,
            "train_noise_stream": TRAIN_NOISE_STREAM, "matched": [list(s) for s in MATCHED],
            "mismatched": [list(s) for s in MISMATCHED]}


def _ms(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_report(raw: dict[str, Any]) -> str:
    R, ref = raw["results"], raw["results"][REFERENCE]
    sets = [key(k, l) for k, l in EVAL_SETS]
    lines = [
        "# RQ2 headroom checks (validation only)",
        "",
        f"Seeds {raw['seeds']}. k-means with {PER_CLASS} patterns per class everywhere; retrieval centre = clean "
        "training mean; β tuned per design on the validation tuning half; numbers on the evaluation half. "
        f"Training noise uses stream {raw['train_noise_stream']} (validation 1, test 0). Robust = mean over "
        + ", ".join(f"{k} {l:g}" for k, l in ROBUST_SETS) + ".",
        "",
        "## 1. Accuracy per design (mean ± std over seeds)",
        "",
        "| Design | β* | " + " | ".join(sets) + " | Robust |",
        "|---|---|" + "---|" * (len(sets) + 1),
    ]
    lines.append("| baseline (no memory) | — | " + " | ".join(_ms([b[s] for b in raw["baseline"]]) for s in sets)
                 + f" | {_ms([b['robust'] for b in raw['baseline']])} |")
    for name, rows in R.items():
        lines.append(f"| {name} | {raw['beta_star'][name]:g} | "
                     + " | ".join(_ms([r[s] for r in rows]) for s in sets) + f" | {_ms([r['robust'] for r in rows])} |")

    lines += ["", f"## 2. Paired differences vs {REFERENCE} (pp; 95 % t-CI)", "",
              "| Design | Δ robust | 95 % CI | +/− | Δ clean | Δ gaussian 0.5 | Δ salt_pepper 0.2 | Δ occlusion 10 |",
              "|---|---|---|---|---|---|---|---|"]
    for name, rows in R.items():
        if name == REFERENCE:
            continue
        s = paired_summary([a["robust"] - b["robust"] for a, b in zip(rows, ref)])

        def d(col: str) -> float:
            return 100 * sum(a[col] - b[col] for a, b in zip(rows, ref)) / len(rows)

        lines.append(f"| {name} | {100 * s['mean']:+.3f} | [{100 * s['ci_low']:+.3f}, {100 * s['ci_high']:+.3f}] | "
                     f"{s['n_pos']}/{s['n_neg']} | {d('clean_0'):+.3f} | {d('gaussian_0.5'):+.3f} | "
                     f"{d('salt_pepper_0.2'):+.3f} | {d('occlusion_10'):+.3f} |")
    lines += ["", "Small-start rows: headroom(n0) = −Δ robust (what an ideal update policy could recover). "
              "Noisy rows: adaptation gain = Δ robust."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("RQ2 headroom checks (validation only).").parse_args(argv)
    cfg = config_from_args(args)
    raw = run_headroom(cfg, retrain=args.retrain)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "memory_headroom"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
