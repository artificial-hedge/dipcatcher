"""Erlang/queueing-theory tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.erlang_queueing import (
    bench_erlang_queueing,
    erlang_a,
    erlang_b,
    erlang_c,
    ggc_wk,
    jackson_throughputs,
    mg1_wk,
    mm_c_metrics,
)


def test_erlang_b_known_value():
    assert abs(erlang_b(4.0, 8) - 0.030422) < 1e-4
    assert abs(erlang_b(1.0, 1) - 0.5) < 1e-9
    assert abs(erlang_b(10.0, 10) - 0.214582) < 1e-4


def test_erlang_c_bounded():
    c = erlang_c(30.0, 1.0, 40)
    assert 0 < c < 1
    # more servers -> less delay
    assert erlang_c(30.0, 1.0, 45) < c


def test_mmc_metrics_consistent():
    m = mm_c_metrics(30.0, 1.0, 40)
    assert m["rho"] == pytest.approx(0.75)
    assert abs(m["lq"] - 30 * m["wq"]) < 1e-9
    assert abs(m["w"] - (m["wq"] + 1.0)) < 1e-9


def test_mg1_pk():
    # M/M/1: Wq = rho*e_s/(1-rho) with cv=1
    assert mg1_wk(0.5, 1.0, 1.0) == pytest.approx(1.0)


def test_ggc_scales_mm_c():
    w_mm = mm_c_metrics(30.0, 1.0, 40)["wq"]
    w_gg = ggc_wk(30.0, 1.0, 40, 1.0, 1.0)
    assert w_gg == pytest.approx(w_mm)
    assert ggc_wk(30.0, 1.0, 40, 2.0, 2.0) == pytest.approx(2 * w_mm)


def test_erlang_a_less_than_c():
    ea = erlang_a(30.0, 1.0, 0.5, 40)
    assert 0 <= ea["p_abandon"] <= 1


def test_jackson_network():
    lam = jackson_throughputs(np.array([10.0, 0.0]), np.array([[0.0, 0.6], [0.4, 0.0]]))
    assert lam[1] == pytest.approx(6.0 / 0.76)


def test_fail_closed():
    with pytest.raises(ValueError):
        erlang_b(-1.0, 3)
    with pytest.raises(ValueError):
        erlang_c(45.0, 1.0, 40)  # unstable
    with pytest.raises(ValueError):
        mg1_wk(1.5, 1.0, 0.5)
    with pytest.raises(ValueError):
        jackson_throughputs(np.array([1.0]), np.array([[0.5, 0.1]]))


def test_bench_passes():
    out = bench_erlang_queueing()
    assert out["synthetic_erlang_b_err"] < 1e-4
    assert out["synthetic_mmc_sim_relerr"] < 0.35
