"""Unit tests for quant_fund.models.dataset_distillation."""

from __future__ import annotations

import pytest

from quant_fund.models.dataset_distillation import bench_dataset_distillation

torch = pytest.importorskip("torch")


def test_inner_steps_changes_result() -> None:
    """inner_steps was a declared-but-unused parameter — the inner
    model never adapted. With it wired in, 0 vs several inner steps
    must produce different distillates."""
    a = bench_dataset_distillation(iters=3, inner_steps=0)
    b = bench_dataset_distillation(iters=3, inner_steps=6)
    assert a["synthetic_dd_acc"] != pytest.approx(b["synthetic_dd_acc"])
