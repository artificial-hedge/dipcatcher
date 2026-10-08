"""Tests for auxiliary_pf — Pitt-Shephard APF."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.auxiliary_pf import (
    auxiliary_pf_sv,
    bench_auxiliary_pf,
    bootstrap_pf_sv,
)


def _sv_path(seed: int = 0, t: int = 100, phi: float = 0.97, sv: float = 0.3):
    rng = np.random.default_rng(seed)
    x = np.zeros(t)
    x[0] = rng.normal(scale=sv / np.sqrt(1 - phi * phi))
    for i in range(1, t):
        x[i] = phi * x[i - 1] + sv * rng.normal()
    y = 0.25 * np.exp(x / 2) * rng.normal(size=t)
    return y


def test_apf_loglik_dominates_bpf():
    y = _sv_path()
    a = auxiliary_pf_sv(y, n_part=250, seed=0)
    b = bootstrap_pf_sv(y, n_part=250, seed=1)
    assert a["loglik"] > b["loglik"]


def test_apf_ess_close_to_bpf():
    y = _sv_path()
    a = auxiliary_pf_sv(y, n_part=250, seed=0)
    b = bootstrap_pf_sv(y, n_part=250, seed=1)
    assert a["mean_ess"] >= 0.6 * b["mean_ess"]


def test_loglik_finite():
    y = _sv_path(seed=2)
    out = auxiliary_pf_sv(y, n_part=200, seed=2)
    assert np.isfinite(out["loglik"])


def test_ess_bounded():
    y = _sv_path(seed=3)
    out = auxiliary_pf_sv(y, n_part=200, seed=3)
    ess = np.asarray(out["ess"])
    assert np.all(ess >= 1.0)
    assert np.all(ess <= 200.0)


def test_deterministic():
    y = _sv_path(seed=4)
    a = auxiliary_pf_sv(y, n_part=150, seed=9)
    b = auxiliary_pf_sv(y, n_part=150, seed=9)
    assert a["loglik"] == b["loglik"]


def test_fail_closed_short():
    with pytest.raises(ValueError):
        auxiliary_pf_sv(np.ones(3))


def test_bench():
    out = bench_auxiliary_pf()
    assert out["synthetic_score"] == 1.0
