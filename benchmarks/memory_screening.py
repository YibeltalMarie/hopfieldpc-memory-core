"""Phase 1 screening of memory-construction designs (validation only, any dataset).

Every design is built from the frozen backbone's clean training embeddings of
one seed and evaluated with the same retrieval (centred, normalized Hopfield
bridge; classifier head unchanged):

    no_memory          the classifier head on z
    class_means        1 pattern per class
    kmeans_k{K}        K centroids per class        (repository default: K = 8)
    facility_location_k{K}  K real embeddings per class (greedy coverage; nested)
    random_k{K}        K random real embeddings per class (control)
    all_train          every training embedding (offline reference)

Protocol, fixed before the first screening run:

* Validation is split into fixed halves (``memory_diagnostics.split_validation``).
  beta is chosen per design on the tuning half (seed-mean score over the config's
  ``beta_selection_sets``, ties -> smaller beta); every reported metric uses the
  evaluation half. The test split is never read.
* Metrics: clean accuracy, robustness AUC per noise type on the config's grids,
  and the primary screening score = mean of the noise AUCs. Clean accuracy is a
  guard, not a target.
* Comparisons are paired by seed against ``kmeans_k8``; they are exploratory
  (no multiplicity correction) and decide only what is worth pursuing.
* Headroom = all_train - kmeans_k8 on the screening score. Decision bands:
  >= 1.0 pp go; 0.5-1.0 pp go with modest expectations; < 0.5 pp report before
  building new methods. all_train is a diagnostic reference, not an upper bound.

    python -m benchmarks.memory_screening --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.memory_diagnostics import paired_summary, split_validation
from benchmarks.memory_selection import nested_prefix, select_class_memory
from benchmarks.mnist_baseline import MnistNet, mean_std, noise_grids, resolve_device, robust_auc
from benchmarks.mnist_prototype_retrieval import HopfieldBridge, build_memory, select_beta
from benchmarks.runs import build_arg_parser, config_from_args, load_data, prepare_run, tune_backbone

BUDGETS = (2, 4, 8, 16)
REFERENCE_DESIGN = "kmeans_k8"
OFFLINE_REFERENCE = "all_train"
EXTRA_BETAS = (2000.0, 5000.0, 10000.0)   # beyond the config grid: large beta ~ nearest stored pattern
GO_PP, MODEST_PP = 1.0, 0.5


def design_names() -> list[str]:
    names = ["no_memory", "class_means"]
    for method in ("kmeans", "facility_location", "random"):
        names += [f"{method}_k{k}" for k in BUDGETS]
    return names + [OFFLINE_REFERENCE]


def build_designs(run: Any, device: torch.device) -> dict[str, dict[str, Any]]:
    """Memory patterns, labels, centre and construction seconds for every memory design."""
    z, y = run.z_train, run.splits.train_labels
    center = z.mean(dim=0)
    designs: dict[str, dict[str, Any]] = {}

    def add(name: str, memory: torch.Tensor, labels: torch.Tensor, secs: float) -> None:
        designs[name] = {"memory": memory.cpu(), "labels": labels.cpu(), "center": center, "build_s": secs}

    start = time.perf_counter()
    memory, labels, _ = build_memory(z, y, per_class=1, center=True, seed=run.seed)
    add("class_means", memory, labels, time.perf_counter() - start)
    for k in BUDGETS:
        start = time.perf_counter()
        memory, labels, _ = build_memory(z, y, per_class=k, center=True, seed=run.seed)
        add(f"kmeans_k{k}", memory, labels, time.perf_counter() - start)
    k_max = max(BUDGETS)
    for method in ("facility_location", "random"):   # nested: one run at k_max gives every budget
        dev = device if method == "facility_location" else torch.device("cpu")
        start = time.perf_counter()
        sel, sel_labels, _ = select_class_memory(z.to(dev), y.to(dev), per_class=k_max, method=method,
                                                 center=center.to(dev), seed=run.seed)
        secs = time.perf_counter() - start
        for k in BUDGETS:
            memory, labels = nested_prefix(sel.cpu(), sel_labels.cpu(), per_class_max=k_max, per_class=k)
            add(f"{method}_k{k}", memory, labels, secs)
    add(OFFLINE_REFERENCE, z.clone(), y.clone(), 0.0)
    return designs


def make_net(run: Any, design: dict[str, Any]) -> MnistNet:
    """Frozen encoder and head shared with the baseline; only the bridge differs (inference only)."""
    bridge = HopfieldBridge(design["memory"], design["labels"], beta=1.0, center=design["center"])
    return MnistNet(run.net.encoder, bridge.to(run.device), run.net.head)


class ValView:
    """Validation embeddings of one seed (clean and every needed noise level), detached from its images.

    Keeps memory bounded on CIFAR-10: the seed's image tensors can be freed once
    the corrupted validation sets have been embedded.
    """

    def __init__(self, run: Any, sets: list[tuple[str, float]]) -> None:
        self.z_val, self.labels = run.z_val, run.splits.val_labels
        self.batch_size, self.device = run.batch_size, run.device
        self._emb = {(k, float(lv)): run.val_embeddings(k, lv) for k, lv in sets if k != "clean" and lv != 0}

    def emb(self, kind: str, level: float) -> torch.Tensor:
        return self.z_val if kind == "clean" or level == 0 else self._emb[(kind, float(level))]


@torch.inference_mode()
def accuracy(net: MnistNet, z: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> float:
    """Head accuracy on embeddings z (logits only; retrieval weights are not collected)."""
    net.eval()
    correct = 0
    for i in range(0, z.shape[0], batch_size):
        logits = net.forward_from_embedding(z[i:i + batch_size].to(device))[0]
        correct += (logits.argmax(dim=-1).cpu() == labels[i:i + batch_size]).sum().item()
    return correct / max(1, z.shape[0])


def evaluate(net: MnistNet, view: ValView, idx: torch.Tensor, grids: dict[str, list[float]]) -> dict[str, Any]:
    """Clean accuracy, per-level accuracy and AUC per noise type on validation rows ``idx``."""
    labels = view.labels[idx]
    clean = accuracy(net, view.z_val[idx], labels, view.batch_size, view.device)
    out: dict[str, Any] = {"clean": clean}
    aucs = []
    for kind, levels in grids.items():
        accs = [clean if level == 0 else accuracy(net, view.emb(kind, level)[idx], labels,
                                                  view.batch_size, view.device) for level in levels]
        out[f"{kind}_acc"] = accs
        out[f"{kind}_auc"] = robust_auc(levels, accs)
        aucs.append(out[f"{kind}_auc"])
    out["noise_auc"] = sum(aucs) / len(aucs)
    return out


def run_screening(cfg: dict[str, Any], *, retrain: bool = False, builder: Any = None,
                  names: list[str] | None = None) -> dict[str, Any]:
    """Screen the designs of ``builder`` (default: this module's designs) under the Phase 1 protocol."""
    device = resolve_device(cfg["device"])
    data = load_data(cfg)
    tune_backbone(cfg, data, device, retrain=retrain)   # MNIST: restores the cached tuned settings
    betas = list(cfg["beta_values"]) + [b for b in EXTRA_BETAS if b not in cfg["beta_values"]]
    sel_sets = cfg["beta_selection_sets"]
    grids = noise_grids(cfg)
    names = list(names) if names is not None else design_names()
    per_seed: list[dict[str, Any]] = []
    curves: dict[str, list[list[float]]] = {n: [] for n in names if n != "no_memory"}
    needed = [(k, float(lv)) for k, lv in sel_sets] + [(k, float(lv)) for k, levels in grids.items() for lv in levels]
    for seed in cfg["seeds"]:
        run = prepare_run(cfg, data, seed, device, retrain=retrain)
        tune, ev = split_validation(run.z_val.shape[0], seed)
        view = ValView(run, needed)
        designs = (builder or build_designs)(run, device)
        nets = {n: make_net(run, d) for n, d in designs.items()}
        for name, net in nets.items():
            curve = []
            for beta in betas:
                net.bridge.set_retrieval(beta=beta)
                accs = [accuracy(net, view.emb(k, lv)[tune], view.labels[tune], view.batch_size, device)
                        for k, lv in sel_sets]
                curve.append(sum(accs) / len(accs))
            curves[name].append(curve)
        per_seed.append({"seed": seed, "view": view, "ev": ev, "nets": nets, "baseline": run.net,
                         "build_s": {n: d["build_s"] for n, d in designs.items()},
                         "sizes": {n: int(d["memory"].shape[0]) for n, d in designs.items()}})
        del run   # frees the seed's image tensors; only embeddings are kept
        print(f"  seed {seed}: {len(designs)} memories built, beta curves on {tune.numel()} tuning images")

    beta_star = {n: select_beta(betas, c)[0] for n, c in curves.items()}
    print("beta*: " + ", ".join(f"{n}={b:g}" for n, b in beta_star.items()))
    results: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    for state in per_seed:
        view, ev = state["view"], state["ev"]
        results["no_memory"].append(evaluate(state["baseline"], view, ev, grids))
        for name, net in state["nets"].items():
            net.bridge.set_retrieval(beta=beta_star[name])
            results[name].append(evaluate(net, view, ev, grids))
        print(f"  seed {state['seed']}: evaluated on {ev.numel()} evaluation images")

    return {
        "dataset": cfg["dataset"], "seeds": list(cfg["seeds"]), "budgets": list(BUDGETS),
        "beta_grid": betas, "beta_star": beta_star, "beta_curves": curves, "grids": grids,
        "beta_selection_sets": [list(s) for s in sel_sets], "results": results,
        "build_s": {n: [s["build_s"][n] for s in per_seed] for n in names if n != "no_memory"},
        "memory_size": {n: per_seed[0]["sizes"][n] for n in names if n != "no_memory"},
    }


def comparisons(raw: dict[str, Any], metric: str = "noise_auc") -> dict[str, dict[str, Any]]:
    """Paired differences (design - kmeans_k8) per seed for every design."""
    ref = [r[metric] for r in raw["results"][REFERENCE_DESIGN]]
    return {n: paired_summary([r[metric] - b for r, b in zip(rows, ref)])
            for n, rows in raw["results"].items() if n != REFERENCE_DESIGN}


def headroom_decision(gap_pp: float) -> str:
    if gap_pp >= GO_PP:
        return "GO: clear headroom for better construction"
    if gap_pp >= MODEST_PP:
        return "GO, modest: some headroom; expected gains are small"
    return "REPORT: little headroom; discuss with the project leader before building new methods"


def _ms(values: list[float], scale: float = 1.0, digits: int = 4) -> str:
    mean, std = mean_std([v * scale for v in values])
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _pp(s: dict[str, Any]) -> str:
    p = "—" if s["wilcoxon_p"] is None else f"{s['wilcoxon_p']:.4f}"
    return (f"{100 * s['mean']:+.2f} [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}] | "
            f"{s['n_pos']}/{s['n_neg']} | {p}")


def build_report(raw: dict[str, Any]) -> str:
    res, names = raw["results"], list(raw["results"])
    kinds = list(raw["grids"])
    lines = [
        f"# Memory-construction screening — {raw['dataset']} (validation only)", "",
        f"Seeds {raw['seeds']}; beta tuned per design on the tuning half (sets {raw['beta_selection_sets']}); "
        "all metrics on the evaluation half. Exploratory: no multiplicity correction. The test split was not read.",
        "Grids: " + "; ".join(f"{k} {v}" for k, v in raw["grids"].items()), "",
        "## 1. Results per design (mean ± std over seeds)", "",
        "| Design | Patterns | β* | Clean acc | " + " | ".join(f"{k} AUC" for k in kinds) + " | Noise AUC (mean) |",
        "|---|---|---|---|" + "---|" * len(kinds) + "---|",
    ]
    for n in names:
        size = raw["memory_size"].get(n, 0)
        beta = raw["beta_star"].get(n)
        lines.append(f"| {n} | {size} | {'—' if beta is None else f'{beta:g}'} | {_ms([r['clean'] for r in res[n]])} | "
                     + " | ".join(_ms([r[f'{k}_auc'] for r in res[n]]) for k in kinds)
                     + f" | {_ms([r['noise_auc'] for r in res[n]])} |")
    edge = [n for n, b in raw["beta_star"].items() if b in (raw["beta_grid"][0], raw["beta_grid"][-1])]
    if edge:
        lines += ["", f"β* at the edge of the grid for: {edge} (interpret with care)."]

    noise, clean = comparisons(raw, "noise_auc"), comparisons(raw, "clean")
    lines += ["", f"## 2. Paired differences vs {REFERENCE_DESIGN} (pp; mean [95 % CI] | +/− seeds | Wilcoxon p)", "",
              "| Design | Noise AUC | Clean accuracy |", "|---|---|---|"]
    for n in names:
        if n != REFERENCE_DESIGN:
            lines.append(f"| {n} | {_pp(noise[n])} | {_pp(clean[n])} |")

    gap = noise[OFFLINE_REFERENCE]
    k16 = noise["kmeans_k16"]
    lines += ["", "## 3. Headroom", "",
              f"* all_train − {REFERENCE_DESIGN} (noise AUC): {100 * gap['mean']:+.2f} pp "
              f"[{100 * gap['ci_low']:+.2f}, {100 * gap['ci_high']:+.2f}], {gap['n_pos']}/{gap['n_neg']} seeds.",
              f"* kmeans_k16 − {REFERENCE_DESIGN}: {100 * k16['mean']:+.2f} pp "
              f"[{100 * k16['ci_low']:+.2f}, {100 * k16['ci_high']:+.2f}].",
              f"* Decision (bands fixed in advance: ≥ {GO_PP} / {MODEST_PP}–{GO_PP} / < {MODEST_PP} pp): "
              f"**{headroom_decision(100 * gap['mean'])}**.", ""]

    lines += ["## 4. Facility location vs class means and k-means (noise AUC, pp; paired by seed)", "",
              "| Budget | FL − class means | FL − k-means | FL − random |", "|---|---|---|---|"]
    for k in raw["budgets"]:
        fl = [r["noise_auc"] for r in res[f"facility_location_k{k}"]]
        row = []
        for other in ("class_means", f"kmeans_k{k}", f"random_k{k}"):
            s = paired_summary([a - r["noise_auc"] for a, r in zip(fl, res[other])])
            row.append(f"{100 * s['mean']:+.2f} [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}] "
                       f"({s['n_pos']}/{s['n_neg']})")
        lines.append(f"| {k} | " + " | ".join(row) + " |")

    lines += ["", "## 5. Construction time (seconds per seed, mean)", "",
              "| Design | Seconds |", "|---|---|"]
    for n, secs in raw["build_s"].items():
        lines.append(f"| {n} | {sum(secs) / len(secs):.2f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser("Phase 1 memory-construction screening (validation only).").parse_args(argv)
    cfg = config_from_args(args)
    raw = run_screening(cfg, retrain=args.retrain)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "memory_screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    raw["comparisons"] = {"noise_auc": comparisons(raw, "noise_auc"), "clean": comparisons(raw, "clean")}
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
