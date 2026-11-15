"""Stage 1 construction diagnostics (training data and the validation-tune half only).

Three checks of the assumptions behind the proposed construction methods, run on
the cached frozen backbones before any new method is built. Every similarity is
the centred cosine used by retrieval (centre = mean training embedding).

D1  Noise-aware premise. For clean and corrupted validation-tune queries: does a
    same-class clean training embedding lie among the KAPPA nearest ones?
    (hit), is the nearest one same-class (top1), and what fraction of the KAPPA
    neighbours is same-class (purity). "pool" repeats this with one corrupted
    copy of every training image added (training noise stream 2), i.e. the
    reference pool noise-aware construction would see.
D2  Training nearest-pattern accuracy of the k-means memory (k per class).
D3  Cross-class boundary scores B_i (KAPPA neighbours over all classes), the
    same-class density proxy rho_i, their within-class Spearman correlation
    (B vs 1/rho), and B of validation-tune points against the training set
    (encoder-bias check).

Decision rules (docs/phase1_criteria.md, fixed before this file was run):

* D1: noise-aware FL is deprioritised if the hit rate of corrupted queries,
  averaged over the nonzero Gaussian and salt-and-pepper grid levels, is at
  least the hit rate of clean queries. Occlusion is reported, not decisive.
* D2: CA-acc is dropped if the k = 8 training nearest-pattern accuracy > 0.99.
* D3: descriptive; a mean within-class Spearman >= 0.7 flags boundary and
  rarity as likely redundant (the binding check is at the combination stage).

The test split is never read.

    python -m benchmarks.construction_diagnostics --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from benchmarks.memory_diagnostics import split_validation
from benchmarks.mnist_baseline import corruption_seed, encode, mean_std, noise_grids, resolve_device
from benchmarks.mnist_prototype_retrieval import build_memory
from benchmarks.runs import build_arg_parser, config_from_args, load_data, prepare_run, tune_backbone

KAPPA = 20
DECISION_TYPES = ("gaussian", "salt_pepper")      # D1 decision; occlusion is held out
TRAIN_NOISE_TYPES = ("gaussian", "salt_pepper")   # corrupted training copies (pool)
TRAIN_STREAM = 2                                  # 0 = test, 1 = validation, 2 = training
BUDGETS = (2, 4, 8, 16)
DECISION_K = 8
CA_ACC_THRESHOLD = 0.99
REDUNDANCY_RHO = 0.7
QUANTILES = (0.5, 0.9, 0.99)
BLOCK = 1024
EPS = 1e-8


def nan_mean_std(values: list[float]) -> tuple[float, float]:
    """mean_std over the non-NaN values; (nan, nan) if none (e.g. a constant B gives an undefined Spearman)."""
    finite = [float(v) for v in values if not math.isnan(float(v))]
    return (float("nan"), float("nan")) if not finite else mean_std(finite)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def centred_unit(z: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Rows of (z - center) scaled to unit length (norm floored at EPS)."""
    x = z - center
    return x / x.norm(dim=1, keepdim=True).clamp_min(EPS)


@torch.no_grad()
def knn(
    query_u: torch.Tensor,
    ref_u: torch.Tensor,
    k: int,
    *,
    exclude_self: bool = False,
    device: torch.device = torch.device("cpu"),
    block: int = BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(similarities, indices) [Nq, k] of the k most similar ref rows for each query row.

    Rows must be unit vectors, so the dot product is the cosine. exclude_self=True
    means query_u is ref_u and row i may not be its own neighbour. Computed in
    blocks of query rows so memory stays O(block * N_ref).
    """
    if exclude_self and query_u.shape[0] != ref_u.shape[0]:
        raise ValueError("exclude_self requires query_u to be ref_u.")
    available = ref_u.shape[0] - int(exclude_self)
    if not 1 <= k <= available:
        raise ValueError(f"k={k} must be in [1, {available}].")
    ref = ref_u.to(device)
    sims, idxs = [], []
    for start in range(0, query_u.shape[0], block):
        q = query_u[start:start + block].to(device)
        s = q @ ref.T
        if exclude_self:
            rows = torch.arange(q.shape[0], device=device)
            s[rows, rows + start] = float("-inf")
        val, idx = s.topk(k, dim=1)
        sims.append(val.cpu())
        idxs.append(idx.cpu())
    return torch.cat(sims), torch.cat(idxs)


def boundary_scores(
    u_query: torch.Tensor,
    y_query: torch.Tensor,
    u_ref: torch.Tensor,
    y_ref: torch.Tensor,
    kappa: int = KAPPA,
    *,
    same_set: bool,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """B_i = fraction of the kappa most similar reference points with a different label (Eq. 3.8)."""
    _, idx = knn(u_query, u_ref, kappa, exclude_self=same_set, device=device)
    return (y_ref[idx] != y_query[:, None]).float().mean(dim=1)


def density_proxy(
    u: torch.Tensor,
    y: torch.Tensor,
    kappa: int = KAPPA,
    *,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """rho_i = mean coverage similarity (1 + cos) / 2 to the kappa most similar same-class points (Eq. 3.7)."""
    rho = torch.empty(u.shape[0])
    for c in torch.unique(y).tolist():
        members = (y == c).nonzero(as_tuple=True)[0]
        k = min(kappa, members.numel() - 1)
        if k < 1:
            rho[members] = float("nan")
            continue
        sims, _ = knn(u[members], u[members], k, exclude_self=True, device=device)
        rho[members] = ((1.0 + sims) / 2.0).mean(dim=1)
    return rho


def rankdata(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks with ties replaced by their average rank."""
    x = x.double().flatten()
    order = torch.argsort(x, stable=True)
    _, inverse, counts = torch.unique_consecutive(x[order], return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, dim=0).double()
    average = ends - (counts.double() - 1.0) / 2.0
    ranks = torch.empty_like(x)
    ranks[order] = average[inverse]
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation (NaN if either input is constant)."""
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = (ra.norm() * rb.norm()).item()
    return float("nan") if denom == 0 else float((ra @ rb).item() / denom)


def neighbour_stats(
    u_query: torch.Tensor,
    y_query: torch.Tensor,
    u_ref: torch.Tensor,
    y_ref: torch.Tensor,
    kappa: int = KAPPA,
    *,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """hit (any same-class in kappa), top1 (nearest same-class) and purity (same-class fraction)."""
    _, idx = knn(u_query, u_ref, kappa, device=device)
    same = y_ref[idx] == y_query[:, None]
    return {"hit": same.any(dim=1).float().mean().item(),
            "top1": same[:, 0].float().mean().item(),
            "purity": same.float().mean().item()}


def nearest_pattern_accuracy(
    u: torch.Tensor,
    y: torch.Tensor,
    memory_u: torch.Tensor,
    memory_labels: torch.Tensor,
    *,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Fraction of rows whose most similar stored pattern carries their label (Eq. 5.10, A(S))."""
    _, idx = knn(u, memory_u, 1, device=device)
    return (memory_labels[idx[:, 0]] == y).float().mean().item()


# ---------------------------------------------------------------------------
# Corrupted training copies (training noise stream)
# ---------------------------------------------------------------------------


def corrupt_per_image(
    images: torch.Tensor,
    kinds: torch.Tensor,
    levels: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Gaussian (kind 0) or salt-and-pepper (kind 1) noise with a level per image.

    Same semantics as ``mnist_baseline.corrupt_images`` (clamp to [0, 1]; one
    salt-and-pepper draw per pixel applied to every channel), but each image
    has its own type and level. Random numbers are drawn for every image, so
    the result depends only on the generator state, not on the type mix.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape [N, C, H, W], got {tuple(images.shape)}.")
    n, _, height, width = images.shape
    lv = levels.to(images.dtype).view(n, 1, 1, 1)
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
    hit = torch.rand((n, 1, height, width), generator=generator) < lv
    salt = torch.rand((n, 1, height, width), generator=generator) < 0.5
    is_gauss = (kinds == 0).view(n, 1, 1, 1)
    is_sp = (kinds == 1).view(n, 1, 1, 1)
    out = torch.where(is_gauss, (images + noise * lv).clamp(0.0, 1.0), images)
    out = torch.where(is_sp & hit & salt, torch.ones_like(images), out)
    return torch.where(is_sp & hit & ~salt, torch.zeros_like(images), out)


@torch.no_grad()
def corrupted_train_embeddings(run: Any, max_level: dict[str, float], *, chunk: int = 2048) -> torch.Tensor:
    """One corrupted copy of every training image, embedded: type uniform over TRAIN_NOISE_TYPES,
    level uniform on [0, max_level[type]] (the largest calibrated grid level)."""
    g = torch.Generator().manual_seed(corruption_seed(run.seed, "gaussian", 0.0, stream=TRAIN_STREAM))
    images = run.splits.train_images
    n = images.shape[0]
    kinds = torch.randint(0, len(TRAIN_NOISE_TYPES), (n,), generator=g)
    maxima = torch.tensor([float(max_level[t]) for t in TRAIN_NOISE_TYPES])
    levels = torch.rand(n, generator=g) * maxima[kinds]
    chunks = []
    for start in range(0, n, chunk):
        x = corrupt_per_image(images[start:start + chunk], kinds[start:start + chunk],
                              levels[start:start + chunk], g)
        chunks.append(encode(run.net.encoder, x, batch_size=run.batch_size, device=run.device))
    return torch.cat(chunks)


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------


def _summary(values: torch.Tensor) -> dict[str, Any]:
    v = values[~torch.isnan(values)]
    return {"mean": v.mean().item(), "frac_positive": (v > 0).float().mean().item(),
            "quantiles": {str(q): torch.quantile(v.float(), q).item() for q in QUANTILES}}


def diagnose_seed(run: Any, grids: dict[str, list[float]], device: torch.device, kappa: int = KAPPA) -> dict[str, Any]:
    z, y = run.z_train, run.splits.train_labels
    center = z.mean(dim=0)
    u_train = centred_unit(z, center)
    tune, _ = split_validation(run.z_val.shape[0], run.seed)
    y_tune = run.splits.val_labels[tune]
    u_clean = centred_unit(run.z_val[tune], center)

    # D1: noise-aware premise
    z_noisy_train = corrupted_train_embeddings(run, {t: max(grids[t]) for t in TRAIN_NOISE_TYPES})
    u_pool = torch.cat([u_train, centred_unit(z_noisy_train, center)])
    y_pool = torch.cat([y, y])
    d1: dict[str, Any] = {"clean": {**neighbour_stats(u_clean, y_tune, u_train, y, kappa, device=device),
                                    "pool": neighbour_stats(u_clean, y_tune, u_pool, y_pool, kappa, device=device)},
                          "levels": []}
    for kind, levels in grids.items():
        for level in levels:
            if level == 0:
                continue
            u_q = centred_unit(run.val_embeddings(kind, level)[tune], center)
            d1["levels"].append({"kind": kind, "level": float(level),
                                 **neighbour_stats(u_q, y_tune, u_train, y, kappa, device=device),
                                 "pool": neighbour_stats(u_q, y_tune, u_pool, y_pool, kappa, device=device)})

    # D2: training nearest-pattern accuracy of the k-means memory
    d2 = {}
    for k in BUDGETS:
        memory, memory_labels, _ = build_memory(z, y, per_class=k, center=True, seed=run.seed)
        d2[str(k)] = nearest_pattern_accuracy(u_train, y, centred_unit(memory, center), memory_labels,
                                              device=device)

    # D3: boundary, density, overlap, encoder bias
    b_train = boundary_scores(u_train, y, u_train, y, kappa, same_set=True, device=device)
    rho = density_proxy(u_train, y, kappa, device=device)
    inv_rho = 1.0 / rho.clamp_min(EPS)
    per_class = {}
    for c in torch.unique(y).tolist():
        m = y == c
        per_class[str(c)] = {"frac_B_positive": (b_train[m] > 0).float().mean().item(),
                             "spearman_B_inv_rho": spearman(b_train[m], inv_rho[m])}
    b_val = boundary_scores(u_clean, y_tune, u_train, y, kappa, same_set=False, device=device)
    rhos = [v["spearman_B_inv_rho"] for v in per_class.values() if not math.isnan(v["spearman_B_inv_rho"])]
    d3 = {"B_train": _summary(b_train), "B_val_tune": _summary(b_val), "rho": _summary(rho),
          "per_class": per_class,
          "spearman_mean": sum(rhos) / len(rhos) if rhos else float("nan"),
          "spearman_max": max(rhos) if rhos else float("nan")}
    return {"seed": run.seed, "n_train": int(z.shape[0]), "n_tune": int(tune.numel()), "kappa": kappa,
            "d1": d1, "d2": d2, "d3": d3}


# ---------------------------------------------------------------------------
# All seeds, decisions, report
# ---------------------------------------------------------------------------


def run_diagnostics(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = load_data(cfg)
    tune_backbone(cfg, data, device, retrain=retrain)
    grids = noise_grids(cfg)
    per_seed = []
    for seed in cfg["seeds"]:
        run = prepare_run(cfg, data, seed, device, retrain=retrain)
        per_seed.append(diagnose_seed(run, grids, device))
        del run
        print(f"  seed {seed}: diagnostics done")
    return {"dataset": cfg["dataset"], "seeds": list(cfg["seeds"]), "grids": grids, "kappa": KAPPA,
            "per_seed": per_seed}


def decision_hit(seed_row: dict[str, Any]) -> float:
    """Corrupted-query hit rate averaged over the nonzero DECISION_TYPES levels."""
    hits = [r["hit"] for r in seed_row["d1"]["levels"] if r["kind"] in DECISION_TYPES]
    return sum(hits) / len(hits)


def decisions(raw: dict[str, Any]) -> dict[str, Any]:
    rows = raw["per_seed"]
    clean_hit = nan_mean_std([r["d1"]["clean"]["hit"] for r in rows])
    noisy_hit = nan_mean_std([decision_hit(r) for r in rows])
    acc8 = nan_mean_std([r["d2"][str(DECISION_K)] for r in rows])
    rho_mean = nan_mean_std([r["d3"]["spearman_mean"] for r in rows])
    return {
        "d1": {"clean_hit": clean_hit, "corrupted_hit": noisy_hit,
               "triggered": noisy_hit[0] >= clean_hit[0],
               "action": "deprioritise noise-aware FL" if noisy_hit[0] >= clean_hit[0] else "keep noise-aware FL"},
        "d2": {"accuracy_k8": acc8, "triggered": acc8[0] > CA_ACC_THRESHOLD,
               "action": "drop CA-acc" if acc8[0] > CA_ACC_THRESHOLD else "keep CA-acc"},
        "d3": {"spearman_mean": rho_mean, "flag_redundant": rho_mean[0] >= REDUNDANCY_RHO},
    }


def _ms(pair: tuple[float, float], digits: int = 4) -> str:
    return f"{pair[0]:.{digits}f} ± {pair[1]:.{digits}f}"


def build_report(raw: dict[str, Any]) -> str:
    rows, dec = raw["per_seed"], decisions(raw)
    lines = [f"# Stage 1 construction diagnostics — {raw['dataset']} (training + validation-tune only)", "",
             f"Seeds {raw['seeds']}; kappa = {raw['kappa']}; centred cosine; mean ± std over seeds. "
             "The test split was not read.", "", "## Decisions (rules fixed in docs/phase1_criteria.md)", "",
             f"* D1 noise-aware premise: clean hit {_ms(dec['d1']['clean_hit'])}, corrupted hit "
             f"(Gaussian + salt-and-pepper levels) {_ms(dec['d1']['corrupted_hit'])} -> "
             f"**{'TRIGGERED' if dec['d1']['triggered'] else 'not triggered'}: {dec['d1']['action']}**",
             f"* D2 training nearest-pattern accuracy, k-means k = {DECISION_K}: {_ms(dec['d2']['accuracy_k8'])} -> "
             f"**{'TRIGGERED' if dec['d2']['triggered'] else 'not triggered'}: {dec['d2']['action']}**",
             f"* D3 within-class Spearman(B, 1/rho): {_ms(dec['d3']['spearman_mean'], 3)} -> "
             f"{'flag: likely redundant (>= 0.7)' if dec['d3']['flag_redundant'] else 'below 0.7'} (descriptive)",
             "", "## D1. Nearest clean training neighbours of validation-tune queries", "",
             "| Query | hit@kappa | top1 | purity | hit@kappa with noisy pool | top1 with noisy pool |",
             "|---|---|---|---|---|---|"]

    def stat(get) -> str:
        return _ms(nan_mean_std([get(r) for r in rows]))

    lines.append(f"| clean | {stat(lambda r: r['d1']['clean']['hit'])} | {stat(lambda r: r['d1']['clean']['top1'])} | "
                 f"{stat(lambda r: r['d1']['clean']['purity'])} | {stat(lambda r: r['d1']['clean']['pool']['hit'])} | "
                 f"{stat(lambda r: r['d1']['clean']['pool']['top1'])} |")
    for i, lv in enumerate(rows[0]["d1"]["levels"]):
        cells = [stat(lambda r, i=i, key=key: r["d1"]["levels"][i][key]) for key in ("hit", "top1", "purity")]
        cells += [stat(lambda r, i=i, key=key: r["d1"]["levels"][i]["pool"][key]) for key in ("hit", "top1")]
        lines.append(f"| {lv['kind']} {lv['level']:g} | " + " | ".join(cells) + " |")
    lines += ["", "## D2. Training nearest-pattern accuracy of the k-means memory", "",
              "| k per class | accuracy |", "|---|---|"]
    lines += [f"| {k} | {stat(lambda r, k=k: r['d2'][str(k)])} |" for k in BUDGETS]
    lines += ["", "## D3. Boundary scores, density and overlap", "",
              "| Quantity | mean | fraction > 0 | median | 90th pct | 99th pct |", "|---|---|---|---|---|---|"]
    for key, label in (("B_train", "B, training points"), ("B_val_tune", "B, validation-tune vs training"),
                       ("rho", "rho, training points")):
        lines.append(f"| {label} | {stat(lambda r: r['d3'][key]['mean'])} | "
                     f"{stat(lambda r: r['d3'][key]['frac_positive'])} | "
                     + " | ".join(stat(lambda r, q=q: r['d3'][key]['quantiles'][str(q)]) for q in QUANTILES) + " |")
    lines += ["", "| Class | fraction B > 0 | Spearman(B, 1/rho) |", "|---|---|---|"]
    for c in rows[0]["d3"]["per_class"]:
        frac = stat(lambda r: r["d3"]["per_class"][c]["frac_B_positive"])
        vals = [r["d3"]["per_class"][c]["spearman_B_inv_rho"] for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        rho_txt = _ms(nan_mean_std(vals), 3) if vals else "n/a"
        lines.append(f"| {c} | {frac} | {rho_txt} |")
    lines.append(f"\nMax within-class Spearman (seed mean): {stat(lambda r: r['d3']['spearman_max'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("Stage 1 construction diagnostics (training + validation-tune only).").parse_args(argv)
    cfg = config_from_args(args)
    raw = run_diagnostics(cfg, retrain=args.retrain)
    raw["decisions"] = decisions(raw)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "construction_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
