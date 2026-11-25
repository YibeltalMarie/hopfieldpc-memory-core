"""Stage 4 follow-up 2: can the weight on corrupted keys tell clean from corrupted queries, query by query?

Descriptive check before any gated-memory proposal. The candidate gate signal is

    g(query) = total retrieval weight that kv_M_g2_k8 puts on corrupted keys   (0 ... 1)

(memories, k = 8, gamma = 2 and beta* exactly as in Stage 4 and key_value_diagnostics).
For the evaluation-half validation queries (clean and every level of every noise type):

    1. distribution of g: percentiles, share above 0.25 / 0.5 / 0.75, and AUROC of g
       separating each corrupted set from the clean set (1 = perfect, 0.5 = no separation);
    2. accuracy of kmeans_k8 and kv_M_g2_k8 within bins of g: do the queries with high g
       gain from the key-value memory and those with low g lose?

Reading recorded before the run (guidance for whether to propose a gate, not a decision):
separation is "good" if AUROC >= 0.8 for Gaussian and salt-and-pepper at their three
strongest levels and at most 10 % of clean queries have g > 0.5. Any gate threshold read
off these validation data is post hoc and must be confirmed on other data.

    python -m benchmarks.gate_signal_check --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from benchmarks import key_value_diagnostics as kd
from benchmarks import memory_screening as ms
from benchmarks.memory_diagnostics import split_validation
from benchmarks.mnist_baseline import mean_std, noise_grids, resolve_device
from benchmarks.runs import build_arg_parser, config_from_args

GATE_DESIGN = kd.M_NAME
REFERENCE = kd.REFERENCE
THRESHOLDS = (0.25, 0.5, 0.75)
BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0)
PERCENTILES = (10, 25, 50, 75, 90)


def auroc(negative: torch.Tensor, positive: torch.Tensor) -> float:
    """P(positive > negative) + 0.5 P(tie) (Mann-Whitney), via average ranks."""
    x = torch.cat([negative, positive]).double()
    values, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True)
    upper = torch.cumsum(counts, 0).double()
    avg_rank = upper - (counts.double() - 1) / 2            # average 1-based rank of each distinct value
    ranks = avg_rank[inverse]
    n_neg, n_pos = negative.numel(), positive.numel()
    return ((ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def bin_index(g: torch.Tensor) -> torch.Tensor:
    """Bin of each g in BIN_EDGES (the last bin includes 1.0)."""
    edges = torch.tensor(BIN_EDGES[1:-1], dtype=g.dtype)
    return torch.bucketize(g, edges, right=True)


def set_summary(g: torch.Tensor, ok_ref: torch.Tensor, ok_kv: torch.Tensor,
                g_clean: torch.Tensor | None) -> dict[str, Any]:
    q = torch.quantile(g.double(), torch.tensor([p / 100 for p in PERCENTILES], dtype=torch.float64))
    b = bin_index(g)
    bins = []
    for i in range(len(BIN_EDGES) - 1):
        m = b == i
        n = int(m.sum())
        bins.append({"share": n / len(g),
                     "acc_ref": ok_ref[m].double().mean().item() if n else None,
                     "acc_kv": ok_kv[m].double().mean().item() if n else None})
    return {"percentiles": {str(p): v for p, v in zip(PERCENTILES, q.tolist())},
            "above": {f"{t:g}": (g > t).double().mean().item() for t in THRESHOLDS},
            "auroc_vs_clean": None if g_clean is None else auroc(g_clean, g),
            "bins": bins, "acc_ref": ok_ref.double().mean().item(), "acc_kv": ok_kv.double().mean().item()}


def check_seed(run: Any, grids: dict[str, list[float]], beta: dict[str, float], device: torch.device
               ) -> dict[str, Any]:
    designs = kd.build_designs(run, grids, device)
    nets = {n: ms.make_net(run, designs[n]) for n in (REFERENCE, GATE_DESIGN)}
    for n, net in nets.items():
        net.bridge.set_retrieval(beta=beta[n])
    sets = kd.query_sets(grids)
    view = ms.ValView(run, sets)
    _, ev = split_validation(run.z_val.shape[0], run.seed)
    labels = view.labels[ev]
    out: dict[str, Any] = {}
    g_clean = None
    for kind, level in sets:
        z = view.emb(kind, level)[ev]
        st = {n: kd.query_stats(nets[n], z, labels, designs[n]["noisy"], designs[n]["labels"], view.batch_size,
                                device) for n in nets}
        g = st[GATE_DESIGN]["mass_noisy"]
        out[f"{kind}@{level:g}"] = set_summary(g, st[REFERENCE]["correct"], st[GATE_DESIGN]["correct"],
                                               None if kind == "clean" else g_clean)
        if kind == "clean":
            g_clean = g
    return out


def reproduction_gap(per_seed: list[dict[str, Any]], stage4: dict[str, Any], grids: dict[str, list[float]]) -> float:
    gap = 0.0
    for i, seed_out in enumerate(per_seed):
        for n, key in ((REFERENCE, "acc_ref"), (GATE_DESIGN, "acc_kv")):
            ref = stage4["results"][n][i]
            for kind, level in kd.query_sets(grids):
                there = ref["clean"] if kind == "clean" else ref[f"{kind}_acc"][grids[kind].index(level)]
                gap = max(gap, abs(seed_out[f"{kind}@{level:g}"][key] - there))
    return gap


def run_check(cfg: dict[str, Any], stage4: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = ms.load_data(cfg)
    ms.tune_backbone(cfg, data, device, retrain=retrain)
    grids = noise_grids(cfg)
    beta = {n: stage4["beta_star"][n] for n in (REFERENCE, GATE_DESIGN)}
    per_seed = []
    for seed in cfg["seeds"]:
        run = ms.prepare_run(cfg, data, seed, device, retrain=retrain)
        per_seed.append(check_seed(run, grids, beta, device))
        del run
        print(f"  seed {seed}: done")
    return {"dataset": cfg.get("dataset"), "seeds": list(cfg["seeds"]), "beta": beta, "grids": grids,
            "per_seed": per_seed, "reproduction_gap": reproduction_gap(per_seed, stage4, grids)}


def _mean(raw: dict[str, Any], get) -> float | None:
    vals = [v for v in (get(s) for s in raw["per_seed"]) if v is not None]
    return mean_std(vals)[0] if vals else None


def _f(x: float | None, scale: float = 100.0, digits: int = 1) -> str:
    return "—" if x is None else f"{scale * x:.{digits}f}"


def pooled_bin(seed_out: dict[str, Any], members: list[str], i: int) -> tuple[float, float | None, float | None]:
    """Share of queries in bin i and the two accuracies within it, pooled over the query sets ``members``."""
    bins = [seed_out[m]["bins"][i] for m in members]
    total = sum(b["share"] for b in bins)
    if total == 0:
        return 0.0, None, None
    acc_ref = sum(b["share"] * b["acc_ref"] for b in bins if b["share"]) / total
    acc_kv = sum(b["share"] * b["acc_kv"] for b in bins if b["share"]) / total
    return total / len(bins), acc_ref, acc_kv


def build_report(raw: dict[str, Any]) -> str:
    keys = ["clean@0"] + [f"{kind}@{lv:g}" for kind, levels in raw["grids"].items() for lv in levels if lv != 0]
    lines = [f"# Gate-signal check: weight on corrupted keys, query by query — {raw['dataset']} "
             "(validation, descriptive)", "",
             f"Seeds {raw['seeds']}; signal g = weight on corrupted keys in {GATE_DESIGN} (beta "
             f"{raw['beta'][GATE_DESIGN]:g}); reference {REFERENCE} (beta {raw['beta'][REFERENCE]:g}). "
             "Evaluation half; means over seeds; g and shares in %. Reproduction check vs Stage 4: largest "
             f"accuracy difference = {100 * raw['reproduction_gap']:.3f} pp.", "",
             "## 1. Distribution of g and separation from clean queries", "",
             "| Queries | " + " | ".join(f"p{p}" for p in PERCENTILES) + " | "
             + " | ".join(f"g > {t:g}" for t in THRESHOLDS) + " | AUROC vs clean |",
             "|---|" + "---|" * (len(PERCENTILES) + len(THRESHOLDS) + 1)]
    for key in keys:
        pct = [_f(_mean(raw, lambda s, p=p: s[key]["percentiles"][str(p)])) for p in PERCENTILES]
        above = [_f(_mean(raw, lambda s, t=t: s[key]["above"][f"{t:g}"])) for t in THRESHOLDS]
        auc = _mean(raw, lambda s: s[key]["auroc_vs_clean"])
        lines.append(f"| {key} | " + " | ".join(pct + above) + f" | {_f(auc, 1.0, 3)} |")
    headers = [f"g in [{lo:g}, {hi:g}{']' if hi == 1.0 else ')'}" for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
    lines += ["", f"## 2. Accuracy by bin of g: {REFERENCE} vs {GATE_DESIGN}", "",
              "Each cell: share of the queries in the bin / accuracy kmeans / accuracy key-value "
              "(key-value - kmeans in pp). Corrupted rows pool the levels of each type.", "",
              "| Queries | " + " | ".join(headers) + " |", "|---|" + "---|" * (len(BIN_EDGES) - 1)]
    groups = [("clean", ["clean@0"])]
    groups += [(kind, [f"{kind}@{lv:g}" for lv in levels if lv != 0]) for kind, levels in raw["grids"].items()]
    for label, members in groups:
        cells = []
        for i in range(len(BIN_EDGES) - 1):
            share = _mean(raw, lambda s, i=i: pooled_bin(s, members, i)[0])
            acc_ref = _mean(raw, lambda s, i=i: pooled_bin(s, members, i)[1])
            acc_kv = _mean(raw, lambda s, i=i: pooled_bin(s, members, i)[2])
            diff = "—" if acc_ref is None or acc_kv is None else f"{100 * (acc_kv - acc_ref):+.1f}"
            cells.append(f"{_f(share)} / {_f(acc_ref)} / {_f(acc_kv)} ({diff})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser("Stage 4 follow-up 2: gate-signal check (descriptive).")
    parser.add_argument("--stage4-raw", default=None,
                        help="Stage 4 raw.json (default: <reports-dir>/key_value_screening/raw.json)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    stage4_path = Path(args.stage4_raw or Path(cfg["reports_dir"]) / "key_value_screening" / "raw.json")
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    raw = run_check(cfg, stage4, retrain=args.retrain)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "gate_signal_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
