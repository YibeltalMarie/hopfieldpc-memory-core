"""CIFAR-10 architecture pilot (validation only).

Trains each candidate backbone for one seed and records, after every epoch,
validation clean accuracy, the beta-selection score (mean validation
accuracy over ``beta_selection_sets``) and the seconds per epoch. The output
fixes the backbone settings in ``configs/cifar10.yaml`` and gives measured
(not assumed) training-time estimates. The test split is never read.

Candidates are ``name:epochs`` with names from ``PRESETS``; a candidate run
for only one or two epochs serves as a timing measurement.

    python -m benchmarks.cifar10_pilot --config configs/cifar10.yaml \\
        --candidates cnn_small:30 cnn_wide:2 resnet18:1 --reports-dir <DIR>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks import mnist_baseline as mb
from benchmarks.cifar10_baseline import DEFAULT_CONFIG_PATH, load_config
from benchmarks.cifar10_model import augment_batch, build_cifar10_net
from benchmarks.datasets import channel_stats, load_dataset

PRESETS: dict[str, dict[str, Any]] = {
    "cnn_small": {"name": "cnn", "widths": [32, 64, 128], "embed_dim": 64},
    "cnn_wide": {"name": "cnn", "widths": [64, 128, 256], "embed_dim": 64},
    "resnet18": {"name": "resnet18", "embed_dim": 64},
}


def parse_candidates(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in items:
        name, _, epochs = item.partition(":")
        if name not in PRESETS or not epochs.isdigit() or int(epochs) < 1:
            raise ValueError(f"candidate {item!r} must be name:epochs with name in {sorted(PRESETS)}.")
        out.append((name, int(epochs)))
    return out


def val_accuracy(net: Any, images: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> float:
    z = mb.encode(net.encoder, images, batch_size=batch_size, device=device)
    out = mb.forward_embeddings(net, z, batch_size=batch_size, device=device)
    return (out["head_pred"] == labels).float().mean().item()


def pilot_candidate(cfg: dict[str, Any], splits: mb.MnistSplits, model: dict[str, Any], epochs: int,
                    seed: int, device: torch.device) -> dict[str, Any]:
    """Train one candidate; per-epoch validation accuracy, selection score and seconds."""
    mean, std = channel_stats(splits.train_images)
    net = build_cifar10_net(seed, model, mean, std)
    sel_sets = [(k, float(lvl)) for k, lvl in cfg["beta_selection_sets"]]
    val_sets = {(k, lvl): (splits.val_images if k == "clean" or lvl == 0 else
                           mb.make_corrupted_test_set(splits.val_images, seed=seed, kind=k, level=lvl, stream=1))
                for k, lvl in [("clean", 0.0), *sel_sets]}
    epochs_log: list[dict[str, Any]] = []
    mark = {"t": time.perf_counter()}

    def on_epoch_end(epoch: int) -> None:
        train_s = time.perf_counter() - mark["t"]
        accs = {key: val_accuracy(net, x, splits.val_labels, cfg["eval_batch_size"], device)
                for key, x in val_sets.items()}
        epochs_log.append({"epoch": epoch, "train_s": train_s, "val_clean": accs[("clean", 0.0)],
                           "selection_score": sum(accs[key] for key in sel_sets) / len(sel_sets)})
        print(f"    epoch {epoch}: {train_s:.0f} s, val clean {epochs_log[-1]['val_clean']:.4f}, "
              f"selection score {epochs_log[-1]['selection_score']:.4f}")
        mark["t"] = time.perf_counter()

    mb.train_baseline(net, splits.train_images, splits.train_labels, epochs=epochs, batch_size=cfg["batch_size"],
                      lr=cfg["lr"], seed=seed, device=device, on_epoch_end=on_epoch_end,
                      augment=augment_batch if cfg["augment"] else None)
    best = max(epochs_log, key=lambda e: (e["selection_score"], -e["epoch"]))
    return {"model": model, "epochs": epochs, "params": mb.count_trainable_params(net),
            "median_epoch_s": statistics.median(e["train_s"] for e in epochs_log),
            "best_epoch": best["epoch"], "best_selection_score": best["selection_score"],
            "best_val_clean": best["val_clean"], "log": epochs_log}


def build_report(cfg: dict[str, Any], results: dict[str, dict[str, Any]], seed: int, device: str) -> str:
    n_seeds = len(cfg["seeds"])
    lines = [
        "# CIFAR-10 architecture pilot (validation only)", "",
        f"Seed {seed}, device {device}, {cfg['batch_size']} per batch, Adam lr {cfg['lr']}, augment {cfg['augment']}, "
        f"{cfg['train_subset'] or 'all'} training images. Selection score = mean validation accuracy over "
        f"{cfg['beta_selection_sets']}. Estimates scale the median epoch time; they exclude evaluation.", "",
        "| Candidate | Params | s / epoch | Epochs run | Best epoch | Val clean at best | Selection score at best | "
        f"Est. hours for {cfg['epochs']} epochs × {n_seeds} seeds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        hours = r["median_epoch_s"] * cfg["epochs"] * n_seeds / 3600
        lines.append(f"| {name} | {r['params']:,} | {r['median_epoch_s']:.0f} | {r['epochs']} | {r['best_epoch']} | "
                     f"{r['best_val_clean']:.4f} | {r['best_selection_score']:.4f} | {hours:.1f} |")
    lines += ["", "Candidates run for one or two epochs are timing measurements; their accuracy is not comparable."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 architecture pilot (validation only).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--candidates", nargs="+", default=["cnn_small:30", "cnn_wide:2", "resnet18:1"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, quick=args.quick, data_dir=args.data_dir, reports_dir=args.reports_dir)
    candidates = parse_candidates(args.candidates)
    threads = mb.configure_threads(cfg["cost"]["num_threads"])
    device = mb.resolve_device(cfg["device"])
    print(f"Device: {device}  threads: {threads}")
    splits = mb.make_splits(load_dataset("cifar10", cfg["data_dir"]), seed=args.seed, val_size=cfg["val_size"],
                            train_subset=cfg["train_subset"], test_subset=cfg["test_subset"])
    results = {}
    for name, epochs in candidates:
        print(f"  {name}: {epochs} epochs")
        results[name] = pilot_candidate(cfg, splits, PRESETS[name], epochs, args.seed, device)
    report = build_report(cfg, results, args.seed, str(device))
    out_dir = Path(cfg["reports_dir"]) / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    (out_dir / "raw.json").write_text(json.dumps({"config": cfg, "seed": args.seed, "results": results,
                                                  "environment": mb.environment_info(device)}, indent=2, default=str),
                                      encoding="utf-8")
    print()
    print(report)
    print(f"\nReports saved to: {out_dir}")


if __name__ == "__main__":
    main()
