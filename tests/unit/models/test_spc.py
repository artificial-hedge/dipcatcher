"""Tests for spc — control charts + capability."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.spc import (
    bench_spc,
    cusum_chart,
    ewma_chart,
    process_capability,
    xbar_r_chart,
)


def _shifted(seed: int = 0, split: int = 60, n: int = 120, shift: float = 1.5):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(size=split), rng.normal(loc=shift, size=n - split)]), split


def test_xbar_flags_shifted_subgroups():
    x, split = _shifted()
    out = xbar_r_chart(x, subgroup=5, phase1=split // 5)
    assert out["n_ooc_x"] >= 2
    assert out["ucl_x"] > out["center"] > out["lcl_x"]


def test_ewma_signals_after_shift():
    x, split = _shifted()
    out = ewma_chart(x, lam=0.3, phase1=split)
    mask = np.asarray(out["signal_mask"])
    assert mask[split - 2 :].sum() >= 3


def test_ewma_clean_in_control():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    out = ewma_chart(x, lam=0.2, l_width=3.5)
    assert out["n_signals"] <= 2


def test_cusum_signals_after_shift():
    x, split = _shifted(shift=1.2)
    out = cusum_chart(x, phase1=split)
    mask = np.asarray(out["signal_mask"])
    assert mask[split - 5 :].sum() >= 3


def test_capability_capable():
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.3, size=300)
    out = process_capability(x, lsl=-2.0, usl=2.0)
    assert out["cp"] > 1.0 and out["cpk"] > 0.8
    assert out["pct_out"] == pytest.approx(0.0, abs=1.0)


def test_capability_offcenter():
    rng = np.random.default_rng(4)
    x = rng.normal(loc=0.8, size=300)
    out = process_capability(x, lsl=-1.0, usl=2.0)
    assert out["cpk"] < out["cp"]


def test_fail_closed_constant():
    with pytest.raises(ValueError):
        ewma_chart(np.ones(50))


def test_fail_closed_subgroup():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        xbar_r_chart(rng.normal(size=100), subgroup=11)


def test_bench():
    out = bench_spc()
    assert out["synthetic_score"] == 1.0
