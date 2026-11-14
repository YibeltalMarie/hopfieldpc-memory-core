"""Noise-level calibration (validation only).

Measures the frozen backbones' validation accuracy on a dense range of levels
for each noise type, then applies a rule fixed before the measurement
(experiment log, 2026-11-14) to choose the evaluation grids and the
beta-selection levels:

* grid end: the first dense level whose mean validation accuracy is at most
  ``FLOOR`` (0.20); if none is, the largest dense level;
* grid: ``N_LEVELS`` (6) levels from 0 to the end, chosen so that accuracy
  falls in roughly equal steps (each target accuracy is matched to the
  closest dense level, keeping the levels strictly increasing);
* beta-selection level: the dense level, not on the grid, whose accuracy is
  closest to the midpoint between clean and end accuracy.

``robust_auc`` integrates accuracy over the level axis, so the grid end sets
which noise range the AUC measures; the interior levels set how accurately the
trapezoid follows the curve. Cached backbones are required: the script never
trains. The test split is never read.

    python -m benchmarks.noise_calibration --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks import mnist_baseline as mb
from benchmarks.runs import load_config, load_data, prepare_run

FLOOR = 0.20
N_LEVELS = 6
DENSE_LEVELS: dict[str, list[float]] = {
    "gaussian": [round(0.01 * i, 4) for i in range(21)],        # 0.00 ... 0.20
    "salt_pepper": [round(0.005 * i, 4) for i in range(31)],    # 0.000 ... 0.150
    "occlusion": [float(s) for s in range(0, 25, 2)],           # 0 ... 24 px
}


def calibrate_grid(levels: list[float], accs: list[float], *, floor: float = FLOOR,
                   n_levels: int = N_LEVELS) -> dict[str, Any]:
    """Apply the calibration rule to one noise type's dense curve.

    levels: ascending dense levels starting at 0; accs: mean validation accuracy per level.
    Returns {'grid', 'selection_level', 'end_accuracy', 'floor_reached'}.
    """
    if len(levels) != len(accs) or len(levels) < 3:
        raise ValueError("levels and accs must have equal length >= 3.")
    if levels[0] != 0 or any(b <= a for a, b in zip(levels, levels[1:])):
        raise ValueError("levels must start at 0 and be strictly increasing.")
    if n_levels < 2:
        raise ValueError("n_levels must be at least 2.")
    below = [i for i, a in enumerate(accs) if a <= floor]
    end = below[0] if below else len(levels) - 1
    if end < 1:
        raise ValueError("accuracy is already at the floor without noise.")
    targets = [accs[0] + (accs[end] - accs[0]) * k / (n_levels - 1) for k in range(1, n_levels - 1)]
    chosen, prev = [0], 0
    for target in targets:
        candidates = range(prev + 1, end)
        if not candidates:
            break
        best = min(candidates, key=lambda i: (abs(accs[i] - target), i))
        chosen.append(best)
        prev = best
    chosen.append(end)
    mid = (accs[0] + accs[end]) / 2
    off_grid = [i for i in range(1, end) if i not in chosen]
    selection = min(off_grid, key=lambda i: (abs(accs[i] - mid), i)) if off_grid else None
    return {
        "grid": [levels[i] for i in chosen],
        "grid_accuracy": [accs[i] for i in chosen],
        "selection_level": None if selection is None else levels[selection],
        "selection_accuracy": None if selection is None else accs[selection],
        "end_accuracy": accs[end],
        "floor_reached": bool(below),
    }


def yaml_snippet(result: dict[str, dict[str, Any]]) -> str:
    """Config lines for the calibrated grids and beta-selection sets."""
    def fmt(values: list[float], integer: bool = False) -> str:
        return "[" + ", ".join(str(int(v)) if integer else f"{v:g}" for v in values) + "]"

    sel = [["clean", 0]] + [[k, result[k]["selection_level"]] for k in ("gaussian", "salt_pepper")]
    return "\n".join([
        "beta_selection_sets: [" + ", ".join(f'["{k}", {v:g}]' for k, v in sel) + "]",
        f"gaussian_sigmas: {fmt(result['gaussian']['grid'])}",
        f"salt_pepper_probs: {fmt(result['salt_pepper']['grid'])}",
        f"occlusion_sizes: {fmt(result['occlusion']['grid'], integer=True)}",
    ])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Noise-level calibration on validation data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, data_dir=args.data_dir, reports_dir=args.reports_dir)
    missing = [s for s in cfg["seeds"] if not (Path(cfg["reports_dir"]) / f"backbone_seed{s}.pt").exists()]
    if missing:
        raise SystemExit(f"No cached backbones for seeds {missing} in {cfg['reports_dir']}; train them first.")
    device = mb.resolve_device(cfg["device"])
    data = load_data(cfg)
    per_seed: dict[str, list[list[float]]] = {k: [] for k in DENSE_LEVELS}
    for seed in cfg["seeds"]:
        run = prepare_run(cfg, data, seed, device)
        if not run.from_cache:
            raise SystemExit(f"seed {seed}: backbone was retrained (settings changed); calibration aborted.")
        for kind, levels in DENSE_LEVELS.items():
            accs = []
            for level in levels:
                out = mb.forward_embeddings(run.net, run.val_embeddings(kind, level), batch_size=run.batch_size,
                                            device=run.device)
                accs.append((out["head_pred"] == run.splits.val_labels).float().mean().item())
            per_seed[kind].append(accs)
        print(f"  seed {seed}: dense validation curves measured")

    result, lines = {}, ["# Noise calibration (validation only)", "",
                         f"Seeds {cfg['seeds']}; rule: grid ends at the first level with mean validation accuracy "
                         f"<= {FLOOR}, {N_LEVELS} levels in about equal accuracy steps; beta-selection level = "
                         "closest off-grid level to the midpoint accuracy.", ""]
    for kind, levels in DENSE_LEVELS.items():
        mean = [sum(s[i] for s in per_seed[kind]) / len(per_seed[kind]) for i in range(len(levels))]
        result[kind] = {**calibrate_grid(levels, mean), "dense_levels": levels, "dense_mean_accuracy": mean}
        r = result[kind]
        lines += [f"## {kind}", "",
                  "Dense curve: " + ", ".join(f"{lv:g}:{a:.3f}" for lv, a in zip(levels, mean)), "",
                  f"Grid {r['grid']} (accuracy {', '.join(f'{a:.3f}' for a in r['grid_accuracy'])}); "
                  f"floor reached: {r['floor_reached']}; beta-selection level {r['selection_level']} "
                  f"(accuracy {r['selection_accuracy'] if r['selection_accuracy'] is None else round(r['selection_accuracy'], 3)}).", ""]
    snippet = yaml_snippet(result)
    lines += ["## Config lines", "", "```yaml", snippet, "```"]
    text = "\n".join(lines)
    out_dir = Path(cfg["reports_dir"]) / "noise_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps({"seeds": cfg["seeds"], "floor": FLOOR, "n_levels": N_LEVELS,
                                                  "result": result, "per_seed": per_seed}, indent=2),
                                      encoding="utf-8")
    print(text)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
