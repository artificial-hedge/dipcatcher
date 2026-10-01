"""Tests for regression-discontinuity estimation (models/rd.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.rd import (
    bench_rd,
    cct_bandwidth,
    donut_rd,
    fuzzy_rd,
    ik_bandwidth,
    mccrary_test,
    rd_local_linear,
    synth_rd,
)


@pytest.fixture
def sharp():
    return synth_rd(n=1000, tau=1.5, seed=5)


def test_sharp_recovers_tau(sharp):
    out = rd_local_linear(np.asarray(sharp["y"]), np.asarray(sharp["r"]))
    assert abs(float(out["tau"]) - 1.5) < 0.5
    assert float(out["t"]) > 3.0


def test_no_effect_null():
    rng = np.random.default_rng(0)
    r = rng.uniform(-1, 1, 800)
    y = 0.3 * r + 0.4 * rng.standard_normal(800)
    out = rd_local_linear(y, r)
    assert abs(float(out["tau"])) < 0.4


def test_ik_cct_positive(sharp):
    ik = ik_bandwidth(np.asarray(sharp["y"]), np.asarray(sharp["r"]))
    cct = cct_bandwidth(np.asarray(sharp["y"]), np.asarray(sharp["r"]))
    assert 0.001 < float(ik["bw"]) < 2.0
    assert 0.001 < float(cct["bw"]) < float(cct["bw_bias"])


def test_fuzzy_recovers():
    d = synth_rd(n=1500, tau=1.5, fuzzy=True, seed=6)
    fz = fuzzy_rd(np.asarray(d["y"]), np.asarray(d["r"]), np.asarray(d["treat"]))
    assert abs(fz["tau"] - 1.5) < 0.6
    assert 0.3 < fz["first_stage"] < 0.8


def test_fuzzy_requires_first_stage():
    rng = np.random.default_rng(1)
    r = rng.uniform(-1, 1, 400)
    y = rng.standard_normal(400)
    t = np.zeros(400)  # no discontinuity in treatment
    with pytest.raises(ValueError):
        fuzzy_rd(y, r, t, bw=0.5)


def test_mccrary_flags_manipulation():
    clean = synth_rd(n=1500, seed=7)
    manip = synth_rd(n=1500, manipulate=0.8, seed=7)
    z_clean = mccrary_test(np.asarray(clean["r"]))["share_z"]
    z_manip = mccrary_test(np.asarray(manip["r"]))["share_z"]
    assert z_manip > 1.96
    assert abs(z_clean) < 1.96


def test_donut(sharp):
    out = donut_rd(np.asarray(sharp["y"]), np.asarray(sharp["r"]), radius=0.08, bw=0.35)
    assert abs(out["tau"] - 1.5) < 0.7


def test_validation():
    with pytest.raises(ValueError):
        rd_local_linear(np.ones(15), np.ones(15))
    with pytest.raises(ValueError):
        rd_local_linear(np.ones(50), np.ones(50))
    with pytest.raises(ValueError):
        mccrary_test(np.ones(50))
    with pytest.raises(ValueError):
        donut_rd(np.ones(200), np.linspace(-1, 1, 200), radius=0.9)


def test_determinism(sharp):
    a = rd_local_linear(np.asarray(sharp["y"]), np.asarray(sharp["r"]))["tau"]
    b = rd_local_linear(np.asarray(sharp["y"]), np.asarray(sharp["r"]))["tau"]
    assert a == b


def test_bench_keys():
    out = bench_rd()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_tau_err"] < 0.5
    assert out["synthetic_tau_t"] > 3.0
    assert out["synthetic_fuzzy_tau_err"] < 0.6
    assert out["synthetic_mccrary_flags"] == 1.0
    assert out["synthetic_determinism"] == 1.0
