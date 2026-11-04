"""Tests for the RQ2 headroom helpers (benchmarks/memory_headroom.py)."""

from __future__ import annotations

import pytest
import torch

from benchmarks.memory_headroom import first_n_per_class


def test_first_n_per_class_takes_n_of_each_class_deterministically() -> None:
    labels = torch.arange(5).repeat_interleave(20)[torch.randperm(100, generator=torch.Generator().manual_seed(0))]
    rows = first_n_per_class(labels, 7, seed=3)
    assert torch.equal(torch.bincount(labels[rows], minlength=5), torch.full((5,), 7))
    assert len(set(rows.tolist())) == rows.numel()
    assert torch.equal(rows, first_n_per_class(labels, 7, seed=3))
    assert not torch.equal(rows, first_n_per_class(labels, 7, seed=4))


def test_first_n_per_class_is_a_stream_prefix() -> None:
    labels = torch.arange(3).repeat_interleave(30)
    small = set(first_n_per_class(labels, 5, seed=1).tolist())
    large = set(first_n_per_class(labels, 12, seed=1).tolist())
    assert small < large  # the first 5 of the stream are among the first 12


def test_first_n_per_class_rejects_too_large_n0() -> None:
    with pytest.raises(ValueError):
        first_n_per_class(torch.arange(3).repeat_interleave(4), 5, seed=0)
