"""Dataset-aware entry points shared by the memory benchmarks.

A config's ``dataset`` key (default ``"mnist"``) selects the pipeline module
that loads the config, tunes and trains the frozen backbones, and builds the
per-seed ``BaselineRun``. A pipeline module provides::

    load_config(path, *, quick, seed, data_dir, reports_dir) -> cfg
    tune_baseline(cfg, data, device, *, retrain) -> result
    prepare_baseline_run(cfg, data, seed, device, *, retrain) -> BaselineRun

    mnist    benchmarks.mnist_baseline (unchanged behaviour)
    cifar10  benchmarks.cifar10_baseline

Benchmarks call ``config_from_args``, ``load_data``, ``tune_backbone`` and
``prepare_run`` from here, so the same benchmark runs on any registered
dataset by switching its ``--config``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import yaml

from benchmarks import mnist_baseline
from benchmarks.datasets import get_dataset, load_dataset

DEFAULT_DATASET = "mnist"
PIPELINES = {"mnist": "benchmarks.mnist_baseline", "cifar10": "benchmarks.cifar10_baseline"}


def pipeline(name: str) -> ModuleType:
    """The pipeline module of a registered dataset."""
    get_dataset(name)
    if name == "mnist":
        return mnist_baseline
    import importlib

    try:
        return importlib.import_module(PIPELINES[name])
    except ModuleNotFoundError as err:
        if err.name == PIPELINES[name]:
            raise NotImplementedError(f"no pipeline module yet for dataset {name!r} ({PIPELINES[name]}).") from err
        raise


def dataset_of(path: Path) -> str:
    """The ``dataset`` named in a YAML config (default mnist)."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML configuration at {path} must be a mapping, got {type(raw).__name__}.")
    return raw.get("dataset", DEFAULT_DATASET)


def load_config(
    path: Path,
    *,
    quick: bool = False,
    seed: int | None = None,
    data_dir: str | None = None,
    reports_dir: str | None = None,
) -> dict[str, Any]:
    """Validate a config with its dataset's pipeline; cfg['dataset'] is always set."""
    cfg = pipeline(dataset_of(path)).load_config(
        path, quick=quick, seed=seed, data_dir=data_dir, reports_dir=reports_dir
    )
    cfg.setdefault("dataset", DEFAULT_DATASET)
    return cfg


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Same CLI as the MNIST pipeline; ``--config`` decides the dataset."""
    return mnist_baseline.build_arg_parser(description)


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.config, quick=args.quick, seed=args.seed,
                       data_dir=args.data_dir, reports_dir=args.reports_dir)


def load_data(cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(train_images, train_labels, test_images, test_labels) of cfg's dataset."""
    return load_dataset(cfg.get("dataset", DEFAULT_DATASET), cfg["data_dir"])


def tune_backbone(cfg: dict[str, Any], data: Any, device: torch.device, *, retrain: bool = False) -> Any:
    return pipeline(cfg.get("dataset", DEFAULT_DATASET)).tune_baseline(cfg, data, device, retrain=retrain)


def prepare_run(cfg: dict[str, Any], data: Any, seed: int, device: torch.device, *, retrain: bool = False) -> Any:
    return pipeline(cfg.get("dataset", DEFAULT_DATASET)).prepare_baseline_run(
        cfg, data, seed, device, retrain=retrain
    )
