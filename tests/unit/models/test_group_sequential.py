"""Group-sequential boundary tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.group_sequential import (
    alpha_use,
    bench_group_sequential,
    conditional_power,
    gs_boundaries,
)


def test_alpha_use_endpoints():
    au = alpha_use(np.array([0.0, 0.5, 1.0]), 0.05)
    assert au[0] == 0.0
    assert abs(au[-1] - 0.05) < 1e-9
    assert (np.diff(au) >= 0).all()


def test_pocock_flat_boundary():
    out = gs_boundaries(5, alpha=0.05, kind="pocock")
    b = np.asarray(out["bounds"])
    assert np.abs(b - b[0]).max() < 0.15
    assert abs(b[0] - 2.413) < 0.1


def test_obf_decreasing_boundary():
    out = gs_boundaries(5, alpha=0.05, kind="obrien_fleming")
    b = np.asarray(out["bounds"])
    assert (np.diff(b) < 0).all()
    assert b[0] > 4.0


def test_more_looks_raise_boundaries():
    b5 = np.asarray(gs_boundaries(5)["bounds"])
    b10 = np.asarray(gs_boundaries(10)["bounds"])
    assert b5[-1] < 2.1 and b10[-1] > b5[-1]


def test_conditional_power_increases_in_drift():
    obf = gs_boundaries(5, alpha=0.05, kind="obrien_fleming")
    t = np.linspace(0.2, 1.0, 5)
    cp0 = conditional_power(0.5, 2, np.asarray(obf["bounds"]), t, drift=0.0)
    cp3 = conditional_power(0.5, 2, np.asarray(obf["bounds"]), t, drift=3.0)
    assert cp3 > cp0
    assert 0.0 <= cp0 <= 1.0 and 0.0 <= cp3 <= 1.0


def test_input_validation():
    with pytest.raises(ValueError):
        alpha_use(np.array([-0.1]), 0.05)
    with pytest.raises(ValueError):
        gs_boundaries(0)
    with pytest.raises(ValueError):
        conditional_power(1.0, 7, np.zeros(5), np.linspace(0.2, 1, 5))


def test_bench_passes():
    out = bench_group_sequential()
    assert out["synthetic_pocock_max_err"] < 0.1
    assert out["synthetic_obf_exit_err"] < 0.01
    assert abs(out["synthetic_realized_alpha"] - 0.05) < 0.02
    assert out["synthetic_obf_gt_poc_first"] == 1.0
    assert out["synthetic_score"] == 1.0
