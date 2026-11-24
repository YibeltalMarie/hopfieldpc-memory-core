"""Stage 4 follow-up diagnostic: why does key-value FL lose clean and occlusion accuracy? (descriptive)

Rebuilds four Stage 4 memories at the repository default k = 8 (gamma = 2, the Stage 4
tune-half choice at k = 8) and evaluates them at their Stage 4 beta* (read from the Stage 4
raw.json; nothing is tuned here):

    kmeans_k8           reference (no corrupted keys)
    na_fl_B_g2_k8       the key-value keys, returning themselves
    kv_S_g2_k8          same keys, clean source values
    kv_M_g2_k8          same keys, clean served-mean values (the Stage 4 candidate)

For every validation query of the evaluation half (clean, and every level of Gaussian,
salt-and-pepper and occlusion) it records where the retrieval weight goes:

    mass_noisy          total weight on corrupted keys
    mass_wrong          total weight on keys of another class
    mass_wrong_noisy    total weight on corrupted keys of another class
    top key             right / wrong class x clean / corrupted key

and whether the head is correct. Hypothesis (recorded before the run): clean and occluded
queries that key-value FL gets wrong but k-means (or na_fl_B) gets right are mostly queries
whose largest weight is on a corrupted key of another class ("confident wrong-class value").
Descriptive only; no decision depends on it. The accuracies are first checked against the
Stage 4 raw.json (same backbones, same memories, same beta -> same numbers).

    python -m benchmarks.key_value_diagnostics --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from benchmarks import key_value_screening as kv
from benchmarks import memory_screening as ms
from benchmarks import noise_aware_screening as na
from benchmarks.construction_diagnostics import corrupted_train_embeddings
from benchmarks.memory_diagnostics import split_validation
from benchmarks.memory_selection import key_value_selection, served_clean_means
from benchmarks.mnist_baseline import mean_std, noise_grids, resolve_device
from benchmarks.mnist_prototype_retrieval import build_memory
from benchmarks.runs import build_arg_parser, config_from_args

K = 8
GAMMA = 2.0
REFERENCE = f"kmeans_k{K}"
B_NAME, S_NAME, M_NAME = na.na_name("B", GAMMA, K), kv.kv_name("S", GAMMA, K), kv.kv_name("M", GAMMA, K)
DESIGNS = (REFERENCE, B_NAME, S_NAME, M_NAME)
CATEGORIES = ("right_clean", "right_noisy", "wrong_clean", "wrong_noisy")   # top key: class x key type


def build_designs(run: Any, grids: dict[str, list[float]], device: torch.device) -> dict[str, dict[str, Any]]:
    """The four memories, built exactly as in key_value_screening.make_builder, plus a corrupted-key mask."""
    z, y = run.z_train, run.splits.train_labels
    center = z.mean(dim=0)
    z_noisy = corrupted_train_embeddings(run, {t: max(grids[t]) for t in na.COPY_TYPES})
    classes = torch.unique(y).tolist()
    memory, labels, _ = build_memory(z, y, per_class=K, center=True, seed=run.seed)
    designs = {REFERENCE: {"memory": memory.cpu(), "labels": labels.cpu(), "center": center,
                           "noisy": torch.zeros(memory.shape[0], dtype=torch.bool)}}
    sel = {c: key_value_selection(z[y == c], z_noisy[y == c], K, GAMMA, center, device=device) for c in classes}
    keys = torch.cat([sel[c][0] for c in classes])
    noisy = torch.cat([sel[c][2] for c in classes])
    labels = torch.tensor(classes).repeat_interleave(K)
    values_m = torch.cat([served_clean_means(z[y == c], z_noisy[y == c], sel[c][0], GAMMA, center,
                                             fallback=sel[c][1], device=device) for c in classes])
    base = {"memory": keys, "labels": labels, "center": center, "noisy": noisy}
    designs[B_NAME] = dict(base)
    designs[S_NAME] = {**base, "values": torch.cat([sel[c][1] for c in classes])}
    designs[M_NAME] = {**base, "values": values_m}
    return designs


@torch.inference_mode()
def query_stats(net: Any, z: torch.Tensor, y: torch.Tensor, noisy: torch.Tensor, mem_labels: torch.Tensor,
                batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Per-query correctness and retrieval-weight breakdown (all tensors of length len(y), on CPU)."""
    net.eval()
    fields = ("correct", "mass_noisy", "mass_wrong", "mass_wrong_noisy", "top_noisy", "top_wrong")
    out: dict[str, list[torch.Tensor]] = {k: [] for k in fields}
    for i in range(0, z.shape[0], batch_size):
        logits, _, _, aux = net.forward_from_embedding(z[i:i + batch_size].to(device))
        w = aux["weights"].double().cpu()
        yb = y[i:i + batch_size]
        wrong = mem_labels[None, :] != yb[:, None]
        top = w.argmax(dim=1)
        out["correct"].append(logits.argmax(dim=-1).cpu() == yb)
        out["mass_noisy"].append((w * noisy[None, :]).sum(dim=1))
        out["mass_wrong"].append((w * wrong).sum(dim=1))
        out["mass_wrong_noisy"].append((w * (wrong & noisy[None, :])).sum(dim=1))
        out["top_noisy"].append(noisy[top])
        out["top_wrong"].append(mem_labels[top] != yb)
    return {k: torch.cat(v) for k, v in out.items()}


def category(stats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Index into CATEGORIES of each query's top key."""
    return 2 * stats["top_wrong"].long() + stats["top_noisy"].long()


def summarise(stats: dict[str, torch.Tensor]) -> dict[str, Any]:
    cat = category(stats)
    out = {"acc": stats["correct"].double().mean().item()}
    for k in ("mass_noisy", "mass_wrong", "mass_wrong_noisy"):
        out[k] = stats[k].mean().item()
    out["share"] = {c: (cat == i).double().mean().item() for i, c in enumerate(CATEGORIES)}
    out["acc_by_top"] = {c: (stats["correct"][cat == i].double().mean().item() if (cat == i).any() else None)
                         for i, c in enumerate(CATEGORIES)}
    return out


def lost_queries(ref: dict[str, torch.Tensor], new: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Queries right under ``ref`` and wrong under ``new``: how many, and where ``new`` put its top weight."""
    lost = ref["correct"] & ~new["correct"]
    gained = ~ref["correct"] & new["correct"]
    cat = category(new)[lost]
    n = int(lost.sum())
    return {"lost": n / len(lost), "gained": gained.double().mean().item(),
            "lost_top": {c: ((cat == i).double().mean().item() if n else None) for i, c in enumerate(CATEGORIES)}}


def query_sets(grids: dict[str, list[float]]) -> list[tuple[str, float]]:
    return [("clean", 0.0)] + [(kind, float(lv)) for kind, levels in grids.items() for lv in levels if lv != 0]


def diagnose_seed(run: Any, grids: dict[str, list[float]], beta: dict[str, float], device: torch.device
                  ) -> dict[str, Any]:
    designs = build_designs(run, grids, device)
    nets = {n: ms.make_net(run, d) for n, d in designs.items()}
    for n, net in nets.items():
        net.bridge.set_retrieval(beta=beta[n])
    sets = query_sets(grids)
    view = ms.ValView(run, sets)
    _, ev = split_validation(run.z_val.shape[0], run.seed)
    labels = view.labels[ev]
    out: dict[str, Any] = {"noisy_keys": designs[M_NAME]["noisy"].double().mean().item(), "sets": {}}
    for kind, level in sets:
        z = view.emb(kind, level)[ev]
        stats = {n: query_stats(nets[n], z, labels, designs[n]["noisy"], designs[n]["labels"], view.batch_size,
                                device) for n in DESIGNS}
        out["sets"][f"{kind}@{level:g}"] = {
            "summary": {n: summarise(s) for n, s in stats.items()},
            "lost_vs_kmeans": lost_queries(stats[REFERENCE], stats[M_NAME]),
            "lost_S_vs_B": lost_queries(stats[B_NAME], stats[S_NAME]),
        }
    return out


def reproduction_gap(per_seed: list[dict[str, Any]], stage4: dict[str, Any], grids: dict[str, list[float]]) -> float:
    """Largest |accuracy here - Stage 4 accuracy| over seeds, designs and query sets (0 = exact reproduction)."""
    gap = 0.0
    for i, seed_out in enumerate(per_seed):
        for n in DESIGNS:
            ref = stage4["results"][n][i]
            for kind, level in query_sets(grids):
                here = seed_out["sets"][f"{kind}@{level:g}"]["summary"][n]["acc"]
                there = ref["clean"] if kind == "clean" else ref[f"{kind}_acc"][grids[kind].index(level)]
                gap = max(gap, abs(here - there))
    return gap


def run_diagnostics(cfg: dict[str, Any], stage4: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    device = resolve_device(cfg["device"])
    data = ms.load_data(cfg)
    ms.tune_backbone(cfg, data, device, retrain=retrain)
    grids = noise_grids(cfg)
    beta = {n: stage4["beta_star"][n] for n in DESIGNS}
    per_seed = []
    for seed in cfg["seeds"]:
        run = ms.prepare_run(cfg, data, seed, device, retrain=retrain)
        per_seed.append(diagnose_seed(run, grids, beta, device))
        del run
        print(f"  seed {seed}: done")
    return {"dataset": cfg.get("dataset"), "seeds": list(cfg["seeds"]), "k": K, "gamma": GAMMA, "beta": beta,
            "grids": grids, "per_seed": per_seed, "reproduction_gap": reproduction_gap(per_seed, stage4, grids)}


def _seed_mean(raw: dict[str, Any], get) -> float | None:
    vals = [v for v in (get(s) for s in raw["per_seed"]) if v is not None]
    return mean_std(vals)[0] if vals else None


def _type_mean(raw: dict[str, Any], kind: str, get) -> float | None:
    """Mean over seeds of the mean over the corrupted levels of ``kind`` (or the clean set)."""
    keys = ["clean@0"] if kind == "clean" else [f"{kind}@{lv:g}" for lv in raw["grids"][kind] if lv != 0]

    def per_seed(s: dict[str, Any]) -> float | None:
        vals = [v for v in (get(s["sets"][k]) for k in keys) if v is not None]
        return sum(vals) / len(vals) if vals else None
    return _seed_mean(raw, per_seed)


def _f(x: float | None, scale: float = 100.0, digits: int = 1) -> str:
    return "—" if x is None else f"{scale * x:.{digits}f}"


def build_report(raw: dict[str, Any]) -> str:
    kinds = ["clean"] + list(raw["grids"])
    lines = [f"# Stage 4 follow-up: where does key-value FL's retrieval weight go? — {raw['dataset']} "
             "(validation, descriptive)", "",
             f"Seeds {raw['seeds']}; k = {raw['k']}, gamma = {raw['gamma']:g}; beta from Stage 4: "
             + ", ".join(f"{n} = {b:g}" for n, b in raw["beta"].items()) + ". Evaluation half; corrupted rows "
             "are the mean over the 5 levels of each type; all numbers in %. "
             "Reproduction check vs Stage 4 raw.json: largest accuracy difference = "
             f"{100 * raw['reproduction_gap']:.3f} pp.",
             f"Corrupted keys in the key-value memory: {_f(_seed_mean(raw, lambda s: s['noisy_keys']))} %.", "",
             "## 1. Accuracy and retrieval-weight breakdown", "",
             "| Queries | Design | accuracy | weight on corrupted keys | weight on wrong-class keys | "
             "weight on wrong-class corrupted keys | top key wrong class |", "|---|---|---|---|---|---|---|"]
    for kind in kinds:
        for n in DESIGNS:
            g = (lambda key, n=n: (lambda st: st["summary"][n][key]))
            top_wrong = _type_mean(raw, kind, lambda st, n=n: st["summary"][n]["share"]["wrong_clean"]
                                   + st["summary"][n]["share"]["wrong_noisy"])
            lines.append(f"| {kind} | {n} | {_f(_type_mean(raw, kind, g('acc')))} | "
                         f"{_f(_type_mean(raw, kind, g('mass_noisy')))} | "
                         f"{_f(_type_mean(raw, kind, g('mass_wrong')))} | "
                         f"{_f(_type_mean(raw, kind, g('mass_wrong_noisy')))} | {_f(top_wrong)} |")
    lines += ["", f"## 2. {M_NAME}: share of queries and accuracy by the key with the largest weight", "",
              "| Queries | " + " | ".join(f"{c}: share / accuracy" for c in CATEGORIES) + " |",
              "|---|" + "---|" * len(CATEGORIES)]
    for kind in kinds:
        cells = []
        for c in CATEGORIES:
            share = _type_mean(raw, kind, lambda st, c=c: st["summary"][M_NAME]["share"][c])
            acc = _type_mean(raw, kind, lambda st, c=c: st["summary"][M_NAME]["acc_by_top"][c])
            cells.append(f"{_f(share)} / {_f(acc)}")
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    lines += ["", "## 3. Queries lost: right under the reference, wrong under the key-value memory", "",
              "Columns: % of queries lost, % gained, then where the key-value memory put its largest weight "
              "for the lost queries (% of lost).", "",
              "| Queries | comparison | lost | gained | " + " | ".join(f"lost, top {c}" for c in CATEGORIES) + " |",
              "|---|---|---|---|" + "---|" * len(CATEGORIES)]
    for kind in kinds:
        for key, label in (("lost_vs_kmeans", f"{REFERENCE} -> {M_NAME}"), ("lost_S_vs_B", f"{B_NAME} -> {S_NAME}")):
            lost = _type_mean(raw, kind, lambda st, key=key: st[key]["lost"])
            gained = _type_mean(raw, kind, lambda st, key=key: st[key]["gained"])
            tops = [_type_mean(raw, kind, lambda st, key=key, c=c: st[key]["lost_top"][c]) for c in CATEGORIES]
            lines.append(f"| {kind} | {label} | {_f(lost)} | {_f(gained)} | " + " | ".join(_f(t) for t in tops) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser("Stage 4 follow-up: retrieval-weight diagnostic for key-value FL (descriptive).")
    parser.add_argument("--stage4-raw", default=None,
                        help="Stage 4 raw.json (default: <reports-dir>/key_value_screening/raw.json)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    stage4_path = Path(args.stage4_raw or Path(cfg["reports_dir"]) / "key_value_screening" / "raw.json")
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    raw = run_diagnostics(cfg, stage4, retrain=args.retrain)
    report = build_report(raw)
    out_dir = Path(cfg["reports_dir"]) / "key_value_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps(raw, indent=2, default=float), encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
