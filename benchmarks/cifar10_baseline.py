"""CIFAR-10 pipeline: config, frozen backbones and per-seed runs.

Implements the pipeline interface used by ``benchmarks.runs`` (``load_config``,
``tune_baseline``, ``prepare_baseline_run``), reusing the dataset-agnostic
parts of ``benchmarks.mnist_baseline``: splits, training loop, encoding,
corruptions and the ``BaselineRun`` container.

* Normalization: per-channel mean/std of each seed's training split only,
  stored as encoder buffers (saved with the checkpoint).
* Augmentation (random crop + flip) only while training the backbone; every
  embedding used for memory construction or evaluation is unaugmented.
* Backbone hyperparameters are not grid-tuned here (``baseline_tuning`` must
  be null): they are fixed from the validation-only pilot
  (``benchmarks.cifar10_pilot``) and recorded in ``configs/cifar10.yaml``.
* The command-line report uses validation data only; the test split is
  reserved for frozen confirmatory evaluations.

    python -m benchmarks.cifar10_baseline --config configs/cifar10.yaml --reports-dir <DIR>
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from benchmarks import mnist_baseline as mb
from benchmarks.cifar10_model import augment_batch, build_cifar10_net, validate_model_config
from benchmarks.datasets import channel_stats, get_dataset, load_dataset

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "cifar10.yaml"
DEFAULT_REPORTS_DIR = "reports/cifar10"
CIFAR_KEYS = ("dataset", "model", "augment")
SPEC = get_dataset("cifar10")


def validate_cifar10_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a CIFAR-10 config; shared keys go through the MNIST validator's rules."""
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping, got {type(config).__name__}.")
    if config.get("dataset") != "cifar10":
        raise ValueError(f"dataset must be 'cifar10', got {config.get('dataset')!r}.")
    shared = {k: v for k, v in config.items() if k not in CIFAR_KEYS}
    shared.setdefault("reports_dir", DEFAULT_REPORTS_DIR)
    shared.setdefault("placements", ["embedding"])
    cfg = mb.validate_mnist_config(shared)
    if max(cfg["occlusion_sizes"]) > SPEC.image_size:
        raise ValueError(f"occlusion_sizes must not exceed {SPEC.image_size}.")
    if cfg["placements"] != ["embedding"]:
        raise ValueError("CIFAR-10 supports placements: ['embedding'] only.")
    if cfg["baseline_tuning"] is not None:
        raise ValueError("baseline_tuning must be null for CIFAR-10: fix epochs, lr and batch_size "
                         "with benchmarks.cifar10_pilot on validation data.")
    augment = config.get("augment", True)
    if not isinstance(augment, bool):
        raise TypeError(f"augment must be a genuine boolean, got {augment!r}.")
    cfg.update({"dataset": "cifar10", "model": validate_model_config(config.get("model")), "augment": augment})
    return cfg


def load_config(
    path: Path,
    *,
    quick: bool = False,
    seed: int | None = None,
    data_dir: str | None = None,
    reports_dir: str | None = None,
) -> dict[str, Any]:
    """Read YAML, apply ``--quick`` (``cost`` merges per key) then CLI overrides, and validate."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML configuration at {path} must be a mapping, got {type(raw).__name__}.")
    config = {k: v for k, v in raw.items() if k != "quick"}
    if quick:
        overrides = raw.get("quick") or {}
        if not isinstance(overrides, dict) or "quick" in overrides:
            raise ValueError("quick must be a mapping without a nested quick section.")
        for key, value in overrides.items():
            config[key] = {**(config.get("cost") or {}), **value} if key == "cost" else value
    for key, value in (("data_dir", data_dir), ("reports_dir", reports_dir)):
        if value is not None:
            config[key] = value
    if seed is not None:
        config["seeds"] = [seed]
    cfg = validate_cifar10_config(config)
    cfg["quick"] = quick
    return cfg


def checkpoint_fingerprint(cfg: dict[str, Any], seed: int, num_train: int) -> dict[str, Any]:
    """Settings a cached backbone must match to be reused."""
    return {
        "dataset": "cifar10",
        "seed": seed,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "val_size": cfg["val_size"],
        "num_train": num_train,
        "model": cfg["model"],
        "augment": cfg["augment"],
        "optimizer": "adam",
        "normalization": "train-split channel mean/std",
    }


def get_or_train_baseline(
    cfg: dict[str, Any],
    splits: mb.MnistSplits,
    seed: int,
    device: torch.device,
    *,
    retrain: bool = False,
) -> tuple[mb.MnistNet, float, bool]:
    """Load ``backbone_seed{seed}.pt`` if its fingerprint matches, else train and cache it."""
    path = Path(cfg["reports_dir"]) / f"backbone_seed{seed}.pt"
    fingerprint = checkpoint_fingerprint(cfg, seed, splits.train_images.shape[0])
    mean, std = channel_stats(splits.train_images)
    net = build_cifar10_net(seed, cfg["model"], mean, std)
    if path.exists() and not retrain:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") == fingerprint:
            net.load_state_dict(payload["state_dict"])
            print(f"  seed {seed}: loaded cached backbone {path}")
            return net.to(device).eval(), float(payload["train_time_s"]), True
        print(f"  seed {seed}: cached backbone {path} has different settings; retraining")
    print(f"  seed {seed}: training {cfg['model']['name']} on {splits.train_images.shape[0]} images "
          f"for {cfg['epochs']} epochs (augment={cfg['augment']})")
    train_time_s = mb.train_baseline(
        net, splits.train_images, splits.train_labels, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        lr=cfg["lr"], seed=seed, device=device, augment=augment_batch if cfg["augment"] else None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "fingerprint": fingerprint, "train_time_s": train_time_s}, path)
    return net.eval(), train_time_s, False


def tune_baseline(cfg: dict[str, Any], data: Any, device: torch.device, *, retrain: bool = False) -> None:
    """No grid tuning for CIFAR-10 (the validator guarantees baseline_tuning is null)."""
    return None


def prepare_baseline_run(
    cfg: dict[str, Any],
    data: Any,
    seed: int,
    device: torch.device,
    *,
    retrain: bool = False,
) -> mb.BaselineRun:
    """Split data, train (or load) the frozen backbone, embed train/validation, class means."""
    splits = mb.make_splits(data, seed=seed, val_size=cfg["val_size"], train_subset=cfg["train_subset"],
                            test_subset=cfg["test_subset"])
    net, train_time_s, from_cache = get_or_train_baseline(cfg, splits, seed, device, retrain=retrain)
    start = time.perf_counter()
    z_train = mb.encode(net.encoder, splits.train_images, batch_size=cfg["eval_batch_size"], device=device)
    prototypes, memory_labels = mb.compute_prototypes(z_train, splits.train_labels, SPEC.num_classes)
    prototype_time_s = time.perf_counter() - start
    z_val = mb.encode(net.encoder, splits.val_images, batch_size=cfg["eval_batch_size"], device=device)
    return mb.BaselineRun(
        seed=seed, splits=splits, net=net, train_time_s=train_time_s, from_cache=from_cache,
        prototypes=prototypes, memory_labels=memory_labels, prototype_time_s=prototype_time_s,
        z_train=z_train, z_val=z_val, batch_size=cfg["eval_batch_size"], device=device,
    )


def val_head_accuracy(run: mb.BaselineRun, kind: str, level: float) -> float:
    """Baseline head accuracy on one validation set (noise stream 1)."""
    out = mb.forward_embeddings(run.net, run.val_embeddings(kind, level), batch_size=run.batch_size,
                                device=run.device)
    return (out["head_pred"] == run.splits.val_labels).float().mean().item()


def run_baseline_only(cfg: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    """Train or load every seed's backbone and report validation accuracy (no test data)."""
    threads = mb.configure_threads(cfg["cost"]["num_threads"])
    device = mb.resolve_device(cfg["device"])
    print(f"Device: {device}  threads: {threads}")
    data = load_dataset("cifar10", cfg["data_dir"])
    grids = mb.noise_grids(cfg)
    rows = []
    for seed in cfg["seeds"]:
        run = prepare_baseline_run(cfg, data, seed, device, retrain=retrain)
        row: dict[str, Any] = {"seed": seed, "train_time_s": run.train_time_s, "from_cache": run.from_cache,
                               "clean": val_head_accuracy(run, "clean", 0.0)}
        for kind, levels in grids.items():
            accs = [val_head_accuracy(run, kind, level) for level in levels]
            row[f"{kind}_acc"], row[f"{kind}_auc"] = accs, mb.robust_auc(levels, accs)
        rows.append(row)
        print(f"  seed {seed}: validation clean accuracy {row['clean']:.4f}")

    def ms(values: list[float]) -> str:
        mean, std = mb.mean_std(values)
        return f"{mean:.4f} ± {std:.4f}"

    lines = [
        "# CIFAR-10 baseline (validation only)", "",
        f"Seeds {cfg['seeds']}; model {cfg['model']}; augment {cfg['augment']}; epochs {cfg['epochs']}, "
        f"batch {cfg['batch_size']}, lr {cfg['lr']}. Validation noise uses stream 1; the test split is not read.", "",
        f"Clean accuracy: {ms([r['clean'] for r in rows])}; training time per seed: "
        f"{ms([r['train_time_s'] / 60 for r in rows])} min", "",
        "| Noise | Levels | Accuracy per level (mean over seeds) | AUC |", "|---|---|---|---|",
    ]
    for kind, levels in grids.items():
        per_level = [sum(r[f"{kind}_acc"][i] for r in rows) / len(rows) for i in range(len(levels))]
        lines.append(f"| {kind} | {levels} | {', '.join(f'{a:.3f}' for a in per_level)} | "
                     f"{ms([r[f'{kind}_auc'] for r in rows])} |")
    text = "\n".join(lines)
    print(text)
    out_dir = Path(cfg["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_val_summary.md").write_text(text + "\n", encoding="utf-8")
    (out_dir / "baseline_val_results.json").write_text(
        json.dumps({"config": cfg, "environment": mb.environment_info(device), "rows": rows}, indent=2, default=str),
        encoding="utf-8")
    print(f"Reports saved to: {out_dir}")
    return {"rows": rows, "text": text}


def main(argv: list[str] | None = None) -> None:
    parser = mb.build_arg_parser("CIFAR-10 frozen backbones (validation-only report).")
    parser.set_defaults(config=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, quick=args.quick, seed=args.seed, data_dir=args.data_dir,
                      reports_dir=args.reports_dir)
    run_baseline_only(cfg, retrain=args.retrain)


if __name__ == "__main__":
    main()
