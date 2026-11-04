"""P6b diagnostics: why does real-embedding selection lose to k-means centroids?

Validation data only; the test split is never read. Per seed, the validation
split is divided once into a fixed tuning half (beta selection) and an
evaluation half (every measurement below).

Audit C (2 x 2): which regions are represented x what is stored per region.

    kmeans_mean    k-means regions, region average (= existing k-means)
    kmeans_medoid  k-means regions, the member of each k-means cluster closest to its centroid
    fl_mean        facility-location regions, average of each region
    fl_pick        facility-location regions, the selected real embedding (= existing FL)

k-means cluster membership uses k-means' own rule (nearest centroid, Euclidean,
raw embeddings). "Closest", FL region assignment and all similarities use the
train-mean-centred cosine of retrieval.

Audit B: retrieval concentration, error breakdown, coverage of corrupted
queries, subgroup accuracy (near-boundary, rare, typical).

Margin analysis: margin(z) = max same-class s(z, m) - max other-class s(z, m);
with nearest-pattern retrieval a query is right exactly when margin > 0.
Flip analysis compares pick vs mean designs query by query.

Statistics: paired mean difference, SD, 95 % t-interval, d_z = mean / SD, exact
Wilcoxon p with Holm correction over the eight Audit C contrasts.

Run from the repository root with the reports directory holding the cached backbones:

    python -m benchmarks.memory_diagnostics --reports-dir <REPORTS_DIR>
    python -m benchmarks.memory_diagnostics --quick
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from benchmarks.memory_selection import (
    class_coverage,
    coverage_similarity,
    nested_prefix,
    select_class_memory,
)
from benchmarks.memory_selection_benchmark import holm_adjust
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
PLATEAU_BETAS = (50.0, 100.0, 200.0, 500.0, 1000.0, 10000.0)
QUERY_SETS = (("clean", 0.0), ("gaussian", 0.3), ("gaussian", 0.5),
              ("salt_pepper", 0.1), ("salt_pepper", 0.2), ("occlusion", 10))
ROBUST_SETS = QUERY_SETS[1:]
SUBGROUP_QUANTILE = 0.10
DENSITY_NEIGHBOURS = 10
FRAGILE_MARGIN = 0.02
SPLIT_SALT = 7919
CONTRASTS = (
    ("averaging | k-means regions", "kmeans_mean", "kmeans_medoid"),
    ("averaging | FL regions", "fl_mean", "fl_pick"),
    ("regions | averages", "fl_mean", "kmeans_mean"),
    ("regions | real points", "fl_pick", "kmeans_medoid"),
)
FLIP_PAIRS = (("fl_mean", "fl_pick"), ("kmeans_mean", "kmeans_medoid"))
# Two-sided 95 % Student-t quantiles, df = 1..30 (normal beyond).
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def key(kind: str, level: float) -> str:
    return f"{kind}_{level:g}"


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
    centroid_labels: torch.Tensor, center: torch.Tensor | None, *, own_cluster: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each centroid, a real embedding of its class most similar to it (centred cosine).

    own_cluster=True: candidates are the centroid's k-means members (nearest
    centroid by Euclidean distance on raw embeddings, k-means' own rule); an
    empty cluster falls back to the whole class. own_cluster=False: candidates
    are the whole class. A real point is never used twice.
    Returns (patterns [K, D], source rows into z [K]) in centroid order.
    """
    sources: list[int] = []
    for c in centroid_labels.unique(sorted=True).tolist():
        rows = torch.nonzero(labels == c, as_tuple=True)[0]
        cents = centroids[centroid_labels == c]
        if rows.numel() < cents.shape[0]:
            raise ValueError(f"class {c} has fewer embeddings than centroids.")
        q = coverage_similarity(cents, z[rows], center)  # [k, n_c]
        assign = torch.cdist(z[rows].double(), cents.double()).argmin(dim=1)
        used = torch.zeros(rows.numel(), dtype=torch.bool)
        for j in range(cents.shape[0]):
            allowed = ~used
            if own_cluster and bool((allowed & (assign == j)).any()):
                allowed = allowed & (assign == j)
            scores = torch.where(allowed, q[j], torch.full_like(q[j], -1.0))
            i = int(torch.argmax(scores).item())
            sources.append(int(rows[i].item()))
            used[i] = True
    src = torch.tensor(sources, dtype=torch.long)
    return z[src].clone(), src


def region_means(
    z: torch.Tensor, labels: torch.Tensor, picks: torch.Tensor,
    pick_labels: torch.Tensor, center: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average of the same-class embeddings assigned (by max q) to each pick.

    Returns (means [K, D] in pick order, stats): pick_to_mean = mean centred
    cosine of each pick to its region mean; within_region = mean centred cosine
    of members to their mean; min_region_size.
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


def retrieval_margins(
    z: torch.Tensor, labels: torch.Tensor, memory: torch.Tensor,
    memory_labels: torch.Tensor, center: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(margin, s_true, s_other) per query, float64, with s = centred cosine.

    s_true = max similarity to a stored pattern of the query's class,
    s_other = max similarity to a pattern of any other class.
    """
    s = 2 * coverage_similarity(z, memory, center) - 1  # [N, K]
    same = labels.cpu()[:, None] == memory_labels.cpu()[None, :]
    neg_inf = torch.full_like(s, -math.inf)
    s_true = torch.where(same, s, neg_inf).max(dim=1).values
    s_other = torch.where(~same, s, neg_inf).max(dim=1).values
    return s_true - s_other, s_true, s_other


def paired_summary(diffs: list[float]) -> dict[str, Any]:
    """Mean, SD, 95 % t-interval, d_z and exact Wilcoxon p of paired differences."""
    n = len(diffs)
    mean = sum(diffs) / n
    sd = mean_std(diffs)[1]
    t = T975[n - 2] if 2 <= n <= len(T975) + 1 else 1.96
    half = t * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half,
        "d_z": mean / sd if sd > 0 else float("nan"),
        "n_pos": sum(d > 1e-12 for d in diffs), "n_neg": sum(d < -1e-12 for d in diffs),
        "wilcoxon_p": wilcoxon_signed_rank_p(diffs) if n >= 6 else None,
    }


def flip_counts(right_a: torch.Tensor, right_b: torch.Tensor) -> dict[str, int]:
    """Queries fixed (b wrong -> a right) and broken (b right -> a wrong) going from design b to a."""
    return {"fixed": int((right_a & ~right_b).sum()), "broken": int((~right_a & right_b).sum())}


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
        medoids, src = medoid_representatives(z, y, km, km_labels, mu, own_cluster=True)
        _, src_any = medoid_representatives(z, y, km, km_labels, mu, own_cluster=False)
        picks, pick_labels = nested_prefix(fl_all, fl_all_labels, per_class_max=k_max, per_class=k)
        fl_means, stats = region_means(z, y, picks, pick_labels, mu)
        rnd, rnd_labels, _ = build_memory(z, y, per_class=k, center=True, seed=run.seed, method="random")
        designs[f"kmeans_mean_k{k}"] = (km, km_labels)
        designs[f"kmeans_medoid_k{k}"] = (medoids, km_labels)
        designs[f"fl_mean_k{k}"] = (fl_means, pick_labels)
        designs[f"fl_pick_k{k}"] = (picks, pick_labels)
        designs[f"random_k{k}"] = (rnd, rnd_labels)
        extra[f"fl_pick_k{k}"] = stats
        extra[f"kmeans_medoid_k{k}"] = {"medoid_changed_by_own_cluster": (src != src_any).double().mean().item()}
    return designs, extra, mu


def design_geometry(run: Any, memory: torch.Tensor, labels: torch.Tensor, mu: torch.Tensor) -> dict[str, float]:
    train_cov = class_coverage(run.z_train, run.splits.train_labels, memory, labels, mu)
    # Alignment of each pattern with its class-mean direction (centred cosine).
    class_mean = torch.stack([run.z_train[run.splits.train_labels == c].mean(dim=0) for c in labels.tolist()])
    align = (2 * coverage_similarity(memory, class_mean, mu).diagonal() - 1).mean().item()
    return {
        "train_coverage": train_cov.mean().item(),
        "centred_norm": (memory - mu).norm(dim=1).mean().item(),
        "train_centred_norm": (run.z_train - mu).norm(dim=1).mean().item(),
        "class_mean_alignment": align,
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
    flips: dict[str, list[dict[str, Any]]] = {}
    subgroup_sizes = []
    shift = {key(k, l): [] for k, l in ROBUST_SETS}
    for run, ev, mu, nets in seeds_state:
        print(f"  seed {run.seed}: measuring on {ev.numel()} validation images")
        y = run.splits.val_labels[ev]
        groups = baseline_subgroups(run, ev, mu)
        subgroup_sizes.append({g: int(m.sum()) for g, m in groups.items()})
        z_clean = run.z_val[ev]
        query = {key(k, l): run.val_embeddings(k, l)[ev] for k, l in QUERY_SETS}
        base_pred = {kq: forward_embeddings(run.net, zq, batch_size=run.batch_size,
                                            device=run.device)["head_pred"] for kq, zq in query.items()}
        for kq in shift:
            shift[kq].append((2 * coverage_similarity(query[kq], z_clean, mu).diagonal() - 1).mean().item())
        right: dict[str, dict[str, torch.Tensor]] = {}
        margins: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        for name, (net, memory, labels) in nets.items():
            net.bridge.set_retrieval(beta=beta_star[name], num_updates=1)
            per_set, per_group = {}, {g: {} for g in groups}
            right[name], margins[name] = {}, {}
            for kq, zq in query.items():
                out = forward_embeddings(net, zq, batch_size=run.batch_size, device=run.device)
                stats = retrieval_stats(out["weights"], labels, y, out["head_pred"], base_pred[kq])
                stats["coverage"] = class_coverage(zq, y, memory, labels, mu).mean().item()
                m, s_true, s_other = retrieval_margins(zq, y, memory, labels, mu)
                stats["margin_mean"] = m.mean().item()
                stats["margin_median"] = m.median().item()
                stats["fragile"] = (m.abs() < FRAGILE_MARGIN).double().mean().item()
                stats["margin_acc"] = (m > 0).double().mean().item()
                per_set[kq] = stats
                right[name][kq] = out["head_pred"].cpu() == y
                margins[name][kq] = (m, s_true, s_other)
                for g, mask in groups.items():
                    per_group[g][kq] = (out["head_pred"][mask] == y[mask]).float().mean().item()
            for g, mask in groups.items():
                per_group[g]["clean_coverage"] = class_coverage(z_clean[mask], y[mask], memory, labels, mu).mean().item()
            measures.setdefault(name, []).append({"sets": per_set, "groups": per_group})

        for k in BUDGETS:
            for a, b in FLIP_PAIRS:
                na, nb = f"{a}_k{k}", f"{b}_k{k}"
                rec: dict[str, Any] = {"fixed": 0, "broken": 0, "n_queries": 0,
                                       "flip_margin_b": [], "flip_boundary_or_rare": 0,
                                       "d_margin": 0.0, "d_s_true": 0.0, "d_s_other": 0.0}
                hard = groups["boundary"] | groups["rare"]
                for kq in (key(kk, ll) for kk, ll in ROBUST_SETS):
                    ra, rb = right[na][kq], right[nb][kq]
                    fc = flip_counts(ra, rb)
                    flipped = ra != rb
                    rec["fixed"] += fc["fixed"]
                    rec["broken"] += fc["broken"]
                    rec["n_queries"] += int(ra.numel())
                    rec["flip_margin_b"] += margins[nb][kq][0][flipped].abs().tolist()
                    rec["flip_boundary_or_rare"] += int((flipped & hard).sum())
                    ma, ta, oa = margins[na][kq]
                    mb, tb, ob = margins[nb][kq]
                    rec["d_margin"] += (ma - mb).mean().item() / len(ROBUST_SETS)
                    rec["d_s_true"] += (ta - tb).mean().item() / len(ROBUST_SETS)
                    rec["d_s_other"] += (oa - ob).mean().item() / len(ROBUST_SETS)
                fm = sorted(rec.pop("flip_margin_b"))
                n_flip = rec["fixed"] + rec["broken"]
                rec["n_flipped"] = n_flip
                rec["flip_margin_b_median"] = fm[len(fm) // 2] if fm else float("nan")
                rec["flip_hard_share"] = rec["flip_boundary_or_rare"] / n_flip if n_flip else float("nan")
                flips.setdefault(f"{na} vs {nb}", []).append(rec)

    return {
        "seeds": list(cfg["seeds"]), "budgets": list(BUDGETS), "beta_grid": betas,
        "beta_selection_sets": sel_sets, "query_sets": [list(q) for q in QUERY_SETS],
        "val_curves_tune": curves, "beta_star": beta_star, "geometry": geometry,
        "measures": measures, "flips": flips, "noise_shift_cosine": shift,
        "subgroup_sizes": subgroup_sizes,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def robust_mean(m: dict[str, Any], stat: str) -> float:
    keys = [key(k, l) for k, l in ROBUST_SETS]
    return sum(m["sets"][kq][stat] for kq in keys) / len(keys)


def robust_acc(m: dict[str, Any]) -> float:
    return robust_mean(m, "acc")


def _ms(values: list[float], digits: int = 4) -> str:
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _p(p: float | None) -> str:
    return "—" if p is None else f"{p:.4f}"


def contrast_table(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Audit C contrasts on robust accuracy with Holm correction over all eight."""
    M = raw["measures"]
    rows = []
    for k in raw["budgets"]:
        for label, a, b in CONTRASTS:
            A, B = M[f"{a}_k{k}"], M[f"{b}_k{k}"]
            d = [robust_acc(x) - robust_acc(w) for x, w in zip(A, B)]
            dc = [x["sets"]["clean_0"]["acc"] - w["sets"]["clean_0"]["acc"] for x, w in zip(A, B)]
            dv = [robust_mean(x, "coverage") - robust_mean(w, "coverage") for x, w in zip(A, B)]
            rows.append({"k": k, "label": label, "a": a, "b": b, **paired_summary(d),
                         "clean_delta": sum(dc) / len(dc), "coverage_delta": sum(dv) / len(dv)})
    for row, adj in zip(rows, holm_adjust([r["wilcoxon_p"] for r in rows])):
        row["holm_p"] = adj
    return rows


def build_report(raw: dict[str, Any]) -> str:
    M, G = raw["measures"], raw["geometry"]
    names = list(M)
    keys = [key(k, l) for k, l in QUERY_SETS]
    betas = raw["beta_grid"]
    lines = [
        "# P6b diagnostics (validation only)",
        "",
        f"Seeds {raw['seeds']}. Validation split per seed into a tuning half (β) and an evaluation half "
        f"(all numbers below). β grid {betas} (1e4 ≈ nearest pattern), same rule for every design. "
        "Robust acc = mean head accuracy over " + ", ".join(f"{k} {l:g}" for k, l in ROBUST_SETS) + ".",
        "",
        "## 1. Designs: task, retrieval and geometry (mean ± std over seeds)",
        "",
        "| Design | β* | Clean acc | Robust acc | N_eff (noisy) | Retrieval err (noisy) | Blend err (noisy) "
        "| Train coverage | Centred norm | Class-mean alignment |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        lines.append(
            f"| {n} | {raw['beta_star'][n]:g} | {_ms([m['sets']['clean_0']['acc'] for m in M[n]])} | "
            f"{_ms([robust_acc(m) for m in M[n]])} | {_ms([robust_mean(m, 'n_eff') for m in M[n]], 2)} | "
            f"{_ms([robust_mean(m, 'retrieval_error') for m in M[n]])} | "
            f"{_ms([robust_mean(m, 'blend_error') for m in M[n]])} | "
            f"{_ms([g['train_coverage'] for g in G[n]])} | {_ms([g['centred_norm'] for g in G[n]], 2)} | "
            f"{_ms([g['class_mean_alignment'] for g in G[n]], 4)} |"
        )
    lines += ["", f"Mean centred norm of training embeddings (seed {raw['seeds'][0]}): "
              f"{G[names[0]][0]['train_centred_norm']:.2f}."]
    for k in raw["budgets"]:
        s = G.get(f"fl_pick_k{k}")
        if s:
            lines.append(f"FL k={k}: cosine(pick, its region mean) {_ms([x['pick_to_mean'] for x in s], 3)}; "
                         f"members to region mean {_ms([x['within_region'] for x in s], 3)}; "
                         f"smallest region {min(x['min_region_size'] for x in s):g}.")
        md = G.get(f"kmeans_medoid_k{k}")
        if md:
            lines.append(f"k-means medoid k={k}: share of medoids changed by the own-cluster restriction "
                         f"{_ms([x['medoid_changed_by_own_cluster'] for x in md], 3)}.")

    lines += ["", "## 2. β plateau (seed-mean tuning score)", "",
              "| Design | " + " | ".join(f"β={b:g}" for b in PLATEAU_BETAS) + " | best − β100 (pp) | β1e4 − β100 (pp) |",
              "|---|" + "---|" * (len(PLATEAU_BETAS) + 2)]
    for n in names:
        curves = raw["val_curves_tune"][n]
        mean_curve = [sum(c[i] for c in curves) / len(curves) for i in range(len(betas))]
        at = {b: mean_curve[betas.index(b)] for b in PLATEAU_BETAS if b in betas}
        lines.append(f"| {n} | " + " | ".join(f"{at[b]:.4f}" if b in at else "—" for b in PLATEAU_BETAS)
                     + f" | {100 * (max(mean_curve) - at[100.0]):+.3f} | {100 * (at[10000.0] - at[100.0]):+.3f} |")

    rows = contrast_table(raw)
    lines += ["", "## 3. Audit C contrasts on robust accuracy (paired by seed; Holm over all 8)", "",
              "| k | Contrast | Δ mean (pp) | 95 % CI (pp) | SD (pp) | d_z | +/− | Wilcoxon p | Holm p "
              "| Δ clean (pp) | Δ noisy coverage (×1000) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['k']} | {r['label']} ({r['a']} − {r['b']}) | {100 * r['mean']:+.3f} | "
                     f"[{100 * r['ci_low']:+.3f}, {100 * r['ci_high']:+.3f}] | {100 * r['sd']:.3f} | "
                     f"{r['d_z']:+.2f} | {r['n_pos']}/{r['n_neg']} | {_p(r['wilcoxon_p'])} | {_p(r['holm_p'])} | "
                     f"{100 * r['clean_delta']:+.3f} | {1000 * r['coverage_delta']:+.2f} |")

    lines += ["", "## 4. Margins (noisy sets; margin = best same-class − best other-class centred cosine)", "",
              "| Design | Mean margin | Median margin | Fragile share (|m| < 0.02) | Acc from margin > 0 | Measured acc |",
              "|---|---|---|---|---|---|"]
    for n in names:
        lines.append(f"| {n} | {_ms([robust_mean(m, 'margin_mean') for m in M[n]])} | "
                     f"{_ms([robust_mean(m, 'margin_median') for m in M[n]])} | "
                     f"{_ms([robust_mean(m, 'fragile') for m in M[n]])} | "
                     f"{_ms([robust_mean(m, 'margin_acc') for m in M[n]])} | {_ms([robust_acc(m) for m in M[n]])} |")

    lines += ["", "## 5. Flips from real point to average (noisy sets, summed over sets; mean over seeds)", "",
              "| Pair | Fixed | Broken | Net per 1000 queries | Median |margin| of flipped (before) "
              "| Flipped that are boundary/rare | Δ margin | Δ s_true | Δ s_other |",
              "|---|---|---|---|---|---|---|---|---|"]
    for pair, recs in raw["flips"].items():
        def avg(f: str) -> float:
            vals = [r[f] for r in recs if r[f] == r[f]]
            return sum(vals) / len(vals) if vals else float("nan")
        net = [1000 * (r["fixed"] - r["broken"]) / r["n_queries"] for r in recs]
        lines.append(f"| {pair} | {avg('fixed'):.1f} | {avg('broken'):.1f} | {_ms(net, 2)} | "
                     f"{avg('flip_margin_b_median'):.4f} | {avg('flip_hard_share'):.3f} | "
                     f"{avg('d_margin'):+.5f} | {avg('d_s_true'):+.5f} | {avg('d_s_other'):+.5f} |")

    lines += ["", "## 6. B3: coverage of corrupted queries (mean over seeds)", "",
              "| Design | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for n in names:
        vals = [sum(m["sets"][kq]["coverage"] for m in M[n]) / len(M[n]) for kq in keys]
        lines.append(f"| {n} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
    shift = raw["noise_shift_cosine"]
    lines.append("| cos(noisy, clean) | — | " + " | ".join(f"{sum(v) / len(v):.3f}" for v in shift.values()) + " |")

    sizes = raw["subgroup_sizes"][0]
    rkeys = [key(k, l) for k, l in ROBUST_SETS]
    lines += ["", f"## 7. B4 subgroups (sizes, seed {raw['seeds'][0]}: {sizes}); robust acc and clean coverage", "",
              "| Design | Boundary acc | Rare acc | Typical acc | Boundary cov | Rare cov | Typical cov |",
              "|---|---|---|---|---|---|---|"]
    for n in names:
        row = []
        for stat in ("acc", "cov"):
            for g in ("boundary", "rare", "typical"):
                if stat == "acc":
                    v = [sum(m["groups"][g][kq] for kq in rkeys) / len(rkeys) for m in M[n]]
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
    raw["contrasts"] = contrast_table(raw)
    out_dir = Path(cfg["reports_dir"]) / "memory_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
