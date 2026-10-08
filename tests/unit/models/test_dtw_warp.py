"""Unit tests for quant_fund.models.dtw_warp."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dtw_warp import (
    bench_dtw,
    dtw_cost,
    synth_dtw,
    warp_register,
)


def test_warped_twin_cheaper_than_indep() -> None:
    b, w, i = synth_dtw(seed=1)
    assert dtw_cost(b, w)["cost_per_step"] < dtw_cost(b, i)["cost_per_step"]


def test_path_monotone_endpoints() -> None:
    b, w, _ = synth_dtw(seed=2)
    p = np.asarray(dtw_cost(b, w)["path"])
    assert p[0, 0] == 0 and p[-1, 0] == b.size - 1
    assert np.all(np.diff(p[:, 0]) >= 0) and np.all(np.diff(p[:, 1]) >= 0)


def test_recovers_shift_and_scale() -> None:
    b, w, _ = synth_dtw(seed=3)
    r = warp_register(b, w)
    assert abs(r["shift"] - 7.0) < 4.0
    assert 0.5 < r["scale"] < 2.5


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        dtw_cost(np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        dtw_cost(np.sin(np.arange(50)), np.sin(np.arange(50)), band=0.99)


def test_bench_contract() -> None:
    out = bench_dtw()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_dtw_cost_warped"] < out["synthetic_dtw_cost_indep"]
