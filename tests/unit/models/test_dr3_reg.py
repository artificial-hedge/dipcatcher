"""Tests for the DR3 regularizer bench."""

from __future__ import annotations

import numpy as np
import pytest


def test_random_baseline_is_measured():
    torch = pytest.importorskip("torch")
    del torch
    from quant_fund.models.dr3_reg import bench_dr3_reg

    out = bench_dr3_reg()
    base = out["synthetic_dr3_random_reward"]
    # was hardcoded -0.2: a fabricated baseline. A measured random-policy
    # reward is finite and essentially never lands on the literal.
    assert np.isfinite(base)
    assert base != -0.2
