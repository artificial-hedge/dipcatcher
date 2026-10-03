"""Tests for hong_li — density-forecast evaluation."""

import numpy as np
import pytest

from quant_fund.models.hong_li import (
    bench_hong_li,
    hl_m_statistic,
    pit_ar1,
    pit_gaussian,
    synth_hong_li,
)


def test_uniform_pit_passes() -> None:
    rng = np.random.default_rng(7)
    z = rng.uniform(1e-4, 1.0 - 1e-4, 1500)
    r = hl_m_statistic(z)
    assert abs(r["m_stat"]) < 3.0
    assert abs(r["rho1"]) < 0.1


def test_clumped_pit_fails() -> None:
    rng = np.random.default_rng(8)
    z = np.clip(rng.normal(0.5, 0.1, 1500), 1e-4, 1.0 - 1e-4)
    r = hl_m_statistic(z)
    assert r["m_stat"] > 8.0


def test_correct_model_passes_wrong_fails() -> None:
    d = synth_hong_li(seed=3)
    r_ok = hl_m_statistic(np.asarray(d["z_right"]))
    r_bad = hl_m_statistic(np.asarray(d["z_wrong"]))
    assert abs(r_ok["m_stat"]) < 3.0
    assert r_bad["m_stat"] > 8.0
    assert abs(r_bad["rho1"]) > abs(r_ok["rho1"])


def test_pit_shapes() -> None:
    x = np.arange(1.0, 51.0)
    assert pit_gaussian(x, 25.0, 10.0).shape == (50,)
    d = synth_hong_li(seed=11)
    xa = np.asarray(d["x"])
    assert pit_ar1(xa, 0.5, 1.0).shape == (xa.size - 1,)
    z = pit_ar1(xa, 0.5, 1.0)
    assert np.all((z > 0) & (z < 1))


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        hl_m_statistic(np.ones(10) * 0.5)
    with pytest.raises(ValueError):
        hl_m_statistic(np.linspace(0.0, 1.0, 200))  # touches boundary
    with pytest.raises(ValueError):
        pit_ar1(np.ones(50), 1.2, 1.0)
    with pytest.raises(ValueError):
        pit_gaussian(np.ones(50), 0.0, -1.0)


def test_determinism() -> None:
    d = synth_hong_li(seed=9)
    a = hl_m_statistic(np.asarray(d["z_wrong"]))
    b = hl_m_statistic(np.asarray(d["z_wrong"]))
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_hong_li()
    for k in ("m_right", "rho1_right", "m_wrong", "rho1_wrong", "berkowitz_wrong", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
