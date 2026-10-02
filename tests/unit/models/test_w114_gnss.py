"""Tests for wave-114 GNSS canon: gold_code, klobuchar,
allan_variance, strapdown, lambda_method, rtk."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.allan_variance import (
    allan_deviation,
    bench_allan_variance,
    noise_slope,
)
from quant_fund.models.gold_code import acquire, bench_gold_code, gold_code
from quant_fund.models.klobuchar import bench_klobuchar, klobuchar_meters
from quant_fund.models.lambda_method import bench_lambda_method, lambda_ils
from quant_fund.models.rtk import _L1, bench_rtk, dd_position
from quant_fund.models.strapdown import (
    bench_strapdown,
    dcm_from_quat,
    mech_step,
    quat_from_rotvec,
)


def test_gold_code_first_chips():
    c = gold_code(1)
    assert c.shape == (1023,)
    assert set(np.unique(c)) == {-1.0, 1.0}
    assert np.sum(c) == pytest.approx(1.0)  # balanced: 512 of one chip


def test_gold_code_acquire():
    c = gold_code(7)
    sig = np.roll(c, 100)
    off, pmr = acquire(sig, 7)
    assert off == 100
    assert pmr > 2


def test_klobuchar_zenith_le_low():
    a = np.array([2.676e-08, 1.490e-08, -1.192e-07, 0.0])
    b = np.array([8.806e04, -3.277e04, -1.966e05, 1.967e06])
    z = klobuchar_meters(38, -77, 89, 0, 14 * 3600, a, b)
    lo = klobuchar_meters(38, -77, 10, 0, 14 * 3600, a, b)
    assert lo > z > 0


def test_allan_white():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 50_000)
    taus = np.array([0.1, 0.5, 1.0, 5.0])
    adev = allan_deviation(y, 0.01, taus)
    assert noise_slope(adev, taus) < -0.3


def test_strapdown_quat():
    q = quat_from_rotvec(np.array([0.0, 0, np.pi]))
    C = dcm_from_quat(q)
    x_nav = C @ np.array([1.0, 0, 0])
    assert np.linalg.norm(x_nav - np.array([-1.0, 0, 0])) < 1e-9


def test_strapdown_step():
    q = np.array([1.0, 0, 0, 0])
    v = np.zeros(3)
    p = np.array([0.0, 0.0, 0.0])
    q2, v2, p2 = mech_step(q, v, p, np.array([0, 0, 0.01]), np.zeros(3), 0.0, 0.01)
    assert abs(np.linalg.norm(q2) - 1) < 1e-12
    assert q2[3] != q[3]


def test_lambda_recovers():
    rng = np.random.default_rng(3)
    n = 4
    z = np.array([3.0, -2, 7, 1])
    R = rng.normal(size=(n, n))
    Q = R.T @ R + np.eye(n) * 0.02
    a_hat = z + np.linalg.cholesky(Q) @ rng.normal(size=n) * 0.1
    zf, _, _ = lambda_ils(a_hat, np.eye(n), np.linalg.inv(Q))
    assert np.array_equal(zf, z)


def test_rtk_recovers():
    rng = np.random.default_rng(4)
    k = 8
    az = np.linspace(0, 2 * np.pi, k)[:-1]
    el = np.linspace(0.3, 1.2, k - 1)
    sats = np.column_stack(
        [
            20.2e6 * np.cos(el) * np.cos(az),
            20.2e6 * np.cos(el) * np.sin(az),
            20.2e6 * np.sin(el) + 6.37e6,
        ]
    )
    sats = np.vstack([np.array([0.0, 0, 26.57e6]), sats])
    base = np.array([6.37e6, 0, 0])
    rover = base + np.array([100.0, 80, 30])
    kk = sats.shape[0]
    n_t = rng.integers(-10, 10, kk - 1)
    prb = np.linalg.norm(sats - base, axis=1)
    prr = np.linalg.norm(sats - rover, axis=1)
    cpb = prb.copy()
    cpr = prr + _L1 * np.concatenate([[0], n_t])
    pos, nf, _ = dd_position(base, rover + np.array([0.5, 0, 0]), sats, prb, prr, cpb, cpr)
    assert np.abs(nf - n_t).max() == 0
    assert np.linalg.norm(pos - rover) < 0.05


@pytest.mark.parametrize(
    "fn",
    [
        bench_gold_code,
        bench_klobuchar,
        bench_allan_variance,
        bench_strapdown,
        bench_lambda_method,
        bench_rtk,
    ],
    ids=lambda f: f.__name__,
)
def test_w114_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
