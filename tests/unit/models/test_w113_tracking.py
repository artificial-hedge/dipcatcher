"""Tests for wave-113 tracking canon: jonker_volgenant, jpda, phd,
mht, cov_int, tdoa."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cov_int import bench_cov_int, cov_int
from quant_fund.models.jonker_volgenant import bench_jonker_volgenant, jv_assign
from quant_fund.models.jpda import bench_jpda, jpda_update
from quant_fund.models.mht import MHT, Track, bench_mht
from quant_fund.models.phd_filter import GMPHD, bench_phd
from quant_fund.models.tdoa import bench_tdoa, chan_tdoa, tdoa_residual


def test_jv_square_optimal():
    C = np.array([[4.0, 1, 3], [2, 0, 5], [3, 2, 2]])
    a, tot = jv_assign(C)
    assert tot == pytest.approx(5.0)  # 1+2+2
    assert sorted(a.tolist()) == [0, 1, 2]


def test_jv_rectangular():
    rng = np.random.default_rng(0)
    C = rng.uniform(0, 5, (4, 9))
    a, tot = jv_assign(C)
    assert len(set(a.tolist())) == 4


def test_jpda_beta_sums_one():
    x = np.array([0.0, 0.0])
    P = np.eye(2)
    H = np.array([[1.0, 0.0]])
    R = np.eye(1) * 0.1
    zs = [np.array([0.1]), np.array([0.4]), np.array([5.0])]
    _, _, beta = jpda_update(x, P, zs, H, R)
    assert beta.sum() == pytest.approx(1.0)
    assert beta[-1] > 0  # miss probability


def test_phd_extract_count():
    F = np.eye(2)
    Q = np.eye(2) * 0.001
    H = np.array([[1.0, 0]])
    R = np.eye(1) * 0.01
    flt = GMPHD(F, Q, H, R, w_extract=0.3)
    flt.spawn(np.array([0.0, 0.0]), np.eye(2) * 0.1, 0.8)
    flt.predict()
    flt.update([np.array([0.05])])
    assert flt.cardinality() > 0.5
    assert len(flt.extract()) >= 1


def test_mht_two_tracks():
    F = np.array([[1, 1], [0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0]])
    R = np.eye(1) * 0.02
    mht = MHT(F, Q, H, R, k_best=3)
    mht.hyps = [
        (0.0, [Track(np.array([0.0, 0.1]), np.eye(2)), Track(np.array([5.0, 0.1]), np.eye(2))])
    ]
    trs = mht.scan([np.array([0.05]), np.array([5.05])])
    assert len(trs) == 2


def test_cov_int_identical():
    a = np.array([1.0, 2.0])
    A = np.eye(2)
    c, C, w = cov_int(a, A, a, A)
    assert np.abs(c - a).max() < 1e-9
    assert np.trace(C) <= np.trace(A) + 1e-9


def test_cov_int_posdef():
    rng = np.random.default_rng(1)
    A = np.eye(3) + rng.normal(0, 0.1, (3, 3)) ** 2
    B = np.eye(3) * 2
    _, C, w = cov_int(np.zeros(3), A, np.ones(3), B)
    assert np.linalg.eigvalsh(C).min() > 0
    assert 0 <= w <= 1


def test_tdoa_exact():
    R = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])
    tru = np.array([1.7, 2.3])
    d = np.linalg.norm(R - tru, axis=1)
    x = chan_tdoa(R, d[1:] - d[0])
    assert np.linalg.norm(x - tru) < 1e-6
    assert np.abs(tdoa_residual(R, d[1:] - d[0], x)).max() < 1e-6


@pytest.mark.parametrize(
    "fn",
    [bench_jonker_volgenant, bench_jpda, bench_phd, bench_mht, bench_cov_int, bench_tdoa],
    ids=lambda f: f.__name__,
)
def test_w113_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
