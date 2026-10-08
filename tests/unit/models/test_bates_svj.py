"""Tests for bates_svj (wave-58)."""

import numpy as np
import pytest

from quant_fund.models.bates_svj import bates_call, bates_cf, bench_bates


def test_martingale_cf() -> None:
    phi = bates_cf(-1j, 1.0, 0.5, 0.01, 0.04, 2.0, 0.04, 0.3, -0.5, 0.3, -0.05, 0.1)
    assert abs(phi - np.exp(0.01 * 0.5)) < 1e-6


def test_jumps_raise_otm_put() -> None:
    p = dict(s0=1.0, t=0.25, r=0.0, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)
    svj = bates_call(k=0.8, lam=0.5, muj=-0.1, sigj=0.15, **p)
    nj = bates_call(k=0.8, lam=0.0, muj=0.0, sigj=0.0, **p)
    assert svj > nj


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        bates_cf(0.5, -1.0, 0.5, 0.0, 0.04, 2.0, 0.04, 0.3, -0.5, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        bates_call(1.0, -1.0, 0.5, 0.0, 0.04, 2.0, 0.04, 0.3, -0.5, 0.0, 0.0, 0.0)


def test_determinism() -> None:
    a = bates_call(1.0, 0.9, 0.3, 0.0, 0.04, 2.0, 0.04, 0.3, -0.5, 0.2, -0.05, 0.1)
    b = bates_call(1.0, 0.9, 0.3, 0.0, 0.04, 2.0, 0.04, 0.3, -0.5, 0.2, -0.05, 0.1)
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_bates()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["synthetic_score"] == 1.0
