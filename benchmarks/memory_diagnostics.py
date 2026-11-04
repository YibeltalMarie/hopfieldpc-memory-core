"""P6b diagnostics: why does real-embedding selection lose to k-means centroids?

Validation data only; the test split is never read. Per seed, the validation
split is divided once into a fixed tuning half (beta selection) and an
evaluation half (every measurement below).

Audit C (2 x 2): which regions are represented x what is stored per region.

    kmeans_mean    k-means regions, region average (= existing k-means)
    kmeans_medoid  k-means regions, real embedding closest to each centroid
    fl_mean        facility-location regions, average of each region
    fl_pick        facility-location regions, the selected real embedding (= existing FL)

"Closest" and region assignment use the train-mean-centred cosine of retrieval.

Audit B: retrieval concentration (max weight, entropy, N_eff = 1 / sum w^2),
error breakdown by correct-class weight, coverage of corrupted queries, and
accuracy per subgroup (near-boundary, rare, typical; defined from the
no-memory baseline and training data only).

Run from the repository root with the reports directory holding the cached backbones:

    python -m benchmarks.memory_diagnostics --reports-dir <REPORTS_DIR>
    python -m benchmarks.memory_diagnostics --quick
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from benchmarks.memory_selection import (
    class_coverage,
    coverage_similarity,
    nested_prefix,
    select_class_memory,
)
from benchmarks.mnist_baseline import (
    build_arg_parser,
    config_from_args,
    forward_embeddings,
    load_mnist,
    mean_std,
    prepare_baseline_run,
    resolve_device,
    tune_baseline,
)
from benchmarks.mnist_prototype_retrieval import (
    build_hopfield_net,
    build_memory,
    select_beta,
    wilcoxon_signed_rank_p,
)

BUDGETS = (2, 8)
EXTRA_BETAS = (200.0, 500.0, 1000.0, 10000.0)  # 1e4 ~ nearest stored pattern
QUERY_SETS = (("clean", 0.0), ("gaussian", 0.3), ("gaussian", 0.5),
              ("salt_pepper", 0.1), ("salt_pepper", 0.2), ("occlusion", 10))
ROBUST_SETS = QUERY_SETS[1:]
SUBGROUP_QUANTILE = 0.10
DENSITY_NEIGHBOURS = 10
SPLIT_SALT = 7919
CONTRASTS = (
    ("averaging | k-means regions", "kmeans_mean", "kmeans_medoid"),
    ("averaging | FL regions", "fl_mean", "fl_pick"),
    ("regions | averages", "fl_mean", "kmeans_mean"),
    ("regions | real points", "fl_pick", "kmeans_medoid"),
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def split_validation(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed disjoint halves (tune, eval) of range(n), sorted."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + SPLIT_SALT))
    half = n // 2
    return perm[:half].sort().values, perm[half:].sort().values


def medoid_representatives(
    z: torch.Tensor, labels: torch.Tensor, centroids: torch.Tensor,
    centroid_labels: torch.Tensor, center: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each centroid, the most similar unused real embedding of its class.

    Returns (patterns [K, D], source rows into z [K]) in centroid order.
    """
    sources = []
    for c in centroid_labels.unique(sorted=True).tolist():
        rows = torch.nonzero(labels == c, as_tuple=True)[0]
        cents = centroids[centroid_labels == c]
        if rows.numel() < cents.shape[0]:
            raise ValueError(f"class {c} has fewer embeddings than centroids.")
        q = coverage_similarity(cents, z[rows], center)  # [k, n_c]
        for j in range(cents.shape[0]):
            i = int(torch.argmax(q[j]).item())
            sources.append(int(rows[i].item()))
            q[:, i] = -1.0  # never reuse a real point
    src = torch.tensor(sources, dtype=torch.long)
    return z[src].clone(), src


def region_means(
    z: torch.Tensor, labels: torch.Tensor, picks: torch.Tensor,
    pick_labels: torch.Tensor, center: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average of the same-class embeddings assigned (by max q) to each pick.

    Returns (means [K, D] in pick order, stats): stats["pick_to_mean"] is the
    mean centred cosine between each pick and its region mean;
    stats["within_region"] the mean centred cosine of members to their mean.
    """
    means = torch.empty_like(picks)
    pick_to_mean, within, counts = [], [], []
    for c in pick_labels.unique(sorted=True).tolist():
        idx = torch.nonzero(pick_labels == c, as_tuple=True)[0]
        zc = z[labels == c]
        assign = coverage_similarity(zc, picks[idx], center).argmax(dim=1)
        for local, j in enumerate(idx.tolist()):
            members = zc[assign == local]
            mean = members.mean(dim=0) if members.shape[0] > 0 else picks[j]
            means[j] = mean
            counts.append(members.shape[0])
            pick_to_mean.append(2 * coverage_similarity(picks[j:j + 1], mean[None], center).item() - 1)
            if members.shape[0] > 0:
                within.append(2 * coverage_similarity(members, mean[None], center).mean().item() - 1)
    stats = {
        "pick_to_mean": sum(pick_to_mean) / len(pick_to_mean),
        "within_region": sum(within) / max(len(within), 1),
        "min_region_size": float(min(counts)),
    }
    return means, stats


def retrieval_stats(
    weights: torch.Tensor, memory_labels: torch.Tensor, targets: torch.Tensor,
    head_pred: torch.Tensor, baseline_pred: torch.Tensor, num_classes: int = 10,
) -> dict[str, float]:
    """Concentration and error breakdown of one retrieval pass (means over queries)."""
    w = weights.detach().double().cpu()
    memory_labels, targets = memory_labels.cpu(), targets.cpu()
    head_pred, baseline_pred = head_pred.cpu(), baseline_pred.cpu()
    entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
    neff = 1.0 / (w * w).sum(dim=1)
    class_mass = w.new_zeros(w.shape[0], num_classes).index_add_(1, memory_labels, w)
    true_mass = class_mass[torch.arange(w.shape[0]), targets]
    right = head_pred == targets
    base_right = baseline_pred == targets
    majority = true_mass >= 0.5

    def frac(mask: torch.Tensor) -> float:
        return mask.double().mean().item()

    return {
        "acc": frac(right),
        "max_weight": w.max(dim=1).values.mean().item(),
        "entropy": entropy.mean().item(),
        "n_eff": neff.mean().item(),
        "n_meaningful": (w > 0.01).sum(dim=1).double().mean().item(),
        "true_class_mass": true_mass.mean().item(),
        "proto_acc": frac(class_mass.argmax(dim=1) == targets),
        "retrieval_error": frac(~majority & ~right),
        "blend_error": frac(majority & ~right),
        "rescued": frac(~majority & right),
        "helped": frac(~base_right & right),
        "hurt": frac(base_right & ~right),
    }


def subgroups(
    margin: torch.Tensor, density: torch.Tensor, quantile: float = SUBGROUP_QUANTILE,
) -> dict[str, torch.Tensor]:
    """Boolean masks: boundary (low margin), rare (low density), typical (neither)."""
    boundary = margin <= torch.quantile(margin.double(), quantile).item()
    rare = density <= torch.quantile(density.double(), quantile).item()
    return {"boundary": boundary, "rare": rare, "typical": ~(boundary | rare)}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def build_designs(run: Any, device: torch.device) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]],
                                                           dict[str, dict[str, float]], torch.Tensor]:
    z, y = run.z_train, run.splits.train_labels
    mu = z.mean(dim=0)
    designs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    extra: dict[str, dict[str, float]] = {}
    means, means_labels, _ = build_memory(z, y, per_class=1, center=True, seed=run.seed)
    designs["class_means"] = (means, means_labels)
    k_max = max(BUDGETS)
    fl_all, fl_all_labels, _ = select_class_memory(
        z.to(device), y.to(device), per_class=k_max, method="facility_location",
        center=mu.to(device), seed=run.seed,
    )
    fl_all, fl_all_labels = fl_all.cpu(), fl_all_labels.cpu()
    for k in BUDGETS:
        km, km_labels, _ = build_memory(z, y, per_class=k, center=True, seed=run.seed)
        medoids, _ = medoid_representatives(z, y, km, km_labels, mu)
        picks, pick_labels = nested_prefix(fl_all, fl_all_labels, per_class_max=k_max, per_class=k)
        fl_means, stats = region_means(z, y, picks, pick_labels, mu)
        rnd, rnd_labels, _ = build_memory(z, y, per_class=k, center=True, seed=run.seed, method="random")
        designs[f"kmeans_mean_k{k}"] = (km, km_labels)
        designs[f"kmeans_medoid_k{k}"] = (medoids, km_labels)
        designs[f"fl_mean_k{k}"] = (fl_means, pick_labels)
        designs[f"fl_pick_k{k}"] = (picks, pick_labels)
        designs[f"random_k{k}"] = (rnd, rnd_labels)
        extra[f"fl_pick_k{k}"] = stats
    return designs, extra, mu


def design_geometry(run: Any, memory: torch.Tensor, labels: torch.Tensor, mu: torch.Tensor) -> dict[str, float]:
    train_cov = class_coverage(run.z_train, run.splits.train_labels, memory, labels, mu)
    return {
        "train_coverage": train_cov.mean().item(),  # FL objective / |R| on the training reference
        "centred_norm": (memory - mu).norm(dim=1).mean().item(),
        "train_centred_norm": (run.z_train - mu).norm(dim=1).mean().item(),
    }


def tune_score(net: Any, run: Any, sel_sets: list, tune: torch.Tensor) -> float:
    labels = run.splits.val_labels[tune]
    accs = []
    for kind, level in sel_sets:
        z = run.val_embeddings(kind, level)[tune]
        out = forward_embeddings(net, z, batch_size=run.batch_size, device=run.device)
        accs.append((out["head_pred"] == labels).float().mean().item())
    return sum(accs) / len(accs)


def baseline_subgroups(run: Any, ev: torch.Tensor, mu: torch.Tensor) -> dict[str, torch.Tensor]:
    """Subgroups of the clean eval half, from the no-memory head and training data only."""
    z = run.z_val[ev]
    y = run.splits.val_labels[ev]
    with torch.inference_mode():
        probs = run.net.head(z.to(run.device)).softmax(dim=-1).cpu()
    top2 = probs.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    density = torch.empty(z.shape[0], dtype=torch.float64)
    for c in y.unique().tolist():
        rows = y == c
        q = coverage_similarity(z[rows], run.z_train[run.splits.train_labels == c], mu)
        density[rows] = q.topk(DENSITY_NEIGHBOURS, dim=1).values.mean(dim=1)
    return subgroups(margin, density)


def run_diagnostics(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = load_mnist(cfg["data_dir"])
    tune_baseline(cfg, data, device, retrain=retrain)
    betas = list(cfg["beta_values"]) + [b for b in EXTRA_BETAS if b not in cfg["beta_values"]]
    sel_sets = cfg["beta_selection_sets"]

    seeds_state, curves, geometry = [], {}, {}
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        tune, ev = split_validation(run.z_val.shape[0], seed)
        print(f"  seed {seed}: building designs and tuning beta on {tune.numel()} validation images")
        designs, extra, mu = build_designs(run, device)
        nets = {}
        for name, (memory, labels) in designs.items():
            net = build_hopfield_net(run.net, memory, labels, beta=betas[0], center=mu)
            curve = []
            for beta in betas:
                net.bridge.set_retrieval(beta=beta, num_updates=1)
                curve.append(tune_score(net, run, sel_sets, tune))
            curves.setdefault(name, []).append(curve)
            geometry.setdefault(name, []).append({**design_geometry(run, memory, labels, mu),
                                                  **extra.get(name, {})})
            nets[name] = (net, memory, labels)
        seeds_state.append((run, ev, mu, nets))

    beta_star = {name: select_beta(betas, c)[0] for name, c in curves.items()}
    print("beta* (validation tuning half): " + ", ".join(f"{n}={b:g}" for n, b in beta_star.items()))

    measures: dict[str, list[dict[str, Any]]] = {}
    subgroup_sizes = []
    shift = {f"{k}_{l:g}": [] for k, l in QUERY_SETS[1:]}
    for run, ev, mu, nets in seeds_state:
        print(f"  seed {run.seed}: measuring on {ev.numel()} validation images")
        y = run.splits.val_labels[ev]
        groups = baseline_subgroups(run, ev, mu)
        subgroup_sizes.append({g: int(m.sum()) for g, m in groups.items()})
        z_clean = run.z_val[ev]
        query = {f"{k}_{l:g}": run.val_embeddings(k, l)[ev] for k, l in QUERY_SETS}
        base_pred = {key: forward_embeddings(run.net, zq, batch_size=run.batch_size,
                                             device=run.device)["head_pred"] for key, zq in query.items()}
        for key in shift:
            shift[key].append((2 * coverage_similarity(query[key], z_clean, mu).diagonal() - 1).mean().item())
        for name, (net, memory, labels) in nets.items():
            net.bridge.set_retrieval(beta=beta_star[name], num_updates=1)
            per_set, per_group = {}, {g: {} for g in groups}
            for key, zq in query.items():
                out = forward_embeddings(net, zq, batch_size=run.batch_size, device=run.device)
                stats = retrieval_stats(out["weights"], labels, y, out["head_pred"], base_pred[key])
                stats["coverage"] = class_coverage(zq, y, memory, labels, mu).mean().item()
                per_set[key] = stats
                for g, mask in groups.items():
                    per_group[g][key] = (out["head_pred"][mask] == y[mask]).float().mean().item()
            for g, mask in groups.items():
                per_group[g]["clean_coverage"] = class_coverage(z_clean[mask], y[mask], memory, labels, mu).mean().item()
            measures.setdefault(name, []).append({"sets": per_set, "groups": per_group})

    return {
        "seeds": list(cfg["seeds"]), "budgets": list(BUDGETS), "beta_grid": betas,
        "beta_selection_sets": sel_sets, "query_sets": [list(q) for q in QUERY_SETS],
        "val_curves_tune": curves, "beta_star": beta_star, "geometry": geometry,
        "measures": measures, "noise_shift_cosine": shift, "subgroup_sizes": subgroup_sizes,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def robust_acc(m: dict[str, Any]) -> float:
    keys = [f"{k}_{l:g}" for k, l in ROBUST_SETS]
    return sum(m["sets"][key]["acc"] for key in keys) / len(keys)


def robust_mean(m: dict[str, Any], stat: str) -> float:
    keys = [f"{k}_{l:g}" for k, l in ROBUST_SETS]
    return sum(m["sets"][key][stat] for key in keys) / len(keys)


def _ms(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_report(raw: dict[str, Any]) -> str:
    M, G = raw["measures"], raw["geometry"]
    names = list(M)
    keys = [f"{k}_{l:g}" for k, l in QUERY_SETS]
    lines = [
        "# P6b diagnostics (validation only)",
        "",
        f"Seeds {raw['seeds']}. Validation split per seed into a tuning half (β) and an evaluation half "
        f"(all numbers below). β grid {raw['beta_grid']} (1e4 ≈ nearest pattern), same rule for every "
        "design. Robust acc = mean head accuracy over " + ", ".join(f"{k} {l:g}" for k, l in ROBUST_SETS) + ".",
        "p-values are descriptive (uncorrected diagnostics, not claims).",
        "",
        "## 1. Designs: task, retrieval and geometry (mean ± std over seeds)",
        "",
        "| Design | β* | Clean acc | Robust acc | N_eff (noisy) | Max w (noisy) | True-class mass (noisy) "
        "| Retrieval err (noisy) | Blend err (noisy) | Train coverage | Centred norm |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        lines.append(
            f"| {n} | {raw['beta_star'][n]:g} | {_ms([m['sets']['clean_0']['acc'] for m in M[n]])} | "
            f"{_ms([robust_acc(m) for m in M[n]])} | {_ms([robust_mean(m, 'n_eff') for m in M[n]], 2)} | "
            f"{_ms([robust_mean(m, 'max_weight') for m in M[n]], 3)} | "
            f"{_ms([robust_mean(m, 'true_class_mass') for m in M[n]], 3)} | "
            f"{_ms([robust_mean(m, 'retrieval_error') for m in M[n]])} | "
            f"{_ms([robust_mean(m, 'blend_error') for m in M[n]])} | "
            f"{_ms([g['train_coverage'] for g in G[n]])} | {_ms([g['centred_norm'] for g in G[n]], 3)} |"
        )
    tn = G[names[0]][0]["train_centred_norm"]
    lines += ["", f"Mean centred norm of training embeddings (seed {raw['seeds'][0]}): {tn:.3f}."]
    for k in raw["budgets"]:
        stats = G.get(f"fl_pick_k{k}")
        if stats:
            lines.append(f"FL k={k}: cosine(pick, its region mean) {_ms([s['pick_to_mean'] for s in stats], 3)}; "
                         f"members to region mean {_ms([s['within_region'] for s in stats], 3)}; "
                         f"smallest region {min(s['min_region_size'] for s in stats):g}.")

    lines += ["", "## 2. Audit C contrasts (paired by seed)", "",
              "| k | Contrast | Δ robust acc (pp) | +/− seeds | p | Δ clean acc (pp) | Δ noisy coverage (×1000) |",
              "|---|---|---|---|---|---|---|"]
    for k in raw["budgets"]:
        for label, a, b in CONTRASTS:
            A, B = M[f"{a}_k{k}"], M[f"{b}_k{k}"]
            d = [robust_acc(x) - robust_acc(w) for x, w in zip(A, B)]
            dc = [x["sets"]["clean_0"]["acc"] - w["sets"]["clean_0"]["acc"] for x, w in zip(A, B)]
            dv = [robust_mean(x, "coverage") - robust_mean(w, "coverage") for x, w in zip(A, B)]
            p = wilcoxon_signed_rank_p(d) if len(d) >= 6 else None
            lines.append(f"| {k} | {label} ({a} − {b}) | {100 * sum(d) / len(d):+.2f} | "
                         f"{sum(x > 0 for x in d)}/{sum(x < 0 for x in d)} | {'—' if p is None else f'{p:.4f}'} | "
                         f"{100 * sum(dc) / len(dc):+.2f} | {1000 * sum(dv) / len(dv):+.2f} |")

    lines += ["", "## 3. B3: coverage of corrupted queries (mean over seeds)", "",
              "| Design | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for n in names:
        vals = [sum(m["sets"][key]["coverage"] for m in M[n]) / len(M[n]) for key in keys]
        lines.append(f"| {n} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    shift = raw["noise_shift_cosine"]
    lines.append("| cos(noisy, clean) | — | " + " | ".join(f"{sum(v) / len(v):.3f}" for v in shift.values()) + " |")

    lines += ["", "## 4. B1–B2 per query set: accuracy / retrieval error / blend error / helped / hurt (k-means vs FL)", "",
              "| Design | Set | Acc | N_eff | Retrieval err | Blend err | Helped | Hurt |", "|---|---|---|---|---|---|---|---|"]
    for k in raw["budgets"]:
        for n in (f"kmeans_mean_k{k}", f"fl_pick_k{k}"):
            for key in keys:
                vals = {s: sum(m["sets"][key][s] for m in M[n]) / len(M[n])
                        for s in ("acc", "n_eff", "retrieval_error", "blend_error", "helped", "hurt")}
                lines.append(f"| {n} | {key} | {vals['acc']:.4f} | {vals['n_eff']:.2f} | {vals['retrieval_error']:.4f} | "
                             f"{vals['blend_error']:.4f} | {vals['helped']:.4f} | {vals['hurt']:.4f} |")

    sizes = raw["subgroup_sizes"][0]
    lines += ["", f"## 5. B4 subgroups (sizes, seed {raw['seeds'][0]}: {sizes}); robust acc and clean coverage", "",
              "| Design | Boundary acc | Rare acc | Typical acc | Boundary cov | Rare cov | Typical cov |",
              "|---|---|---|---|---|---|---|"]
    rkeys = [f"{k}_{l:g}" for k, l in ROBUST_SETS]
    for n in names:
        row = []
        for stat in ("acc", "cov"):
            for g in ("boundary", "rare", "typical"):
                if stat == "acc":
                    v = [sum(m["groups"][g][key] for key in rkeys) / len(rkeys) for m in M[n]]
                else:
                    v = [m["groups"][g]["clean_coverage"] for m in M[n]]
                row.append(f"{sum(v) / len(v):.4f}")
        lines.append(f"| {n} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("P6b memory diagnostics (validation only).").parse_args(argv)
    cfg = config_from_args(args)
    raw = run_diagnostics(cfg, retrain=args.retrain)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "memory_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
