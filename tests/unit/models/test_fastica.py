"""Tests for fastica — Hyvarinen FastICA."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fastica import bench_fastica, fast_ica


def _mixed(seed: int = 0, n: int = 2000):
    rng = np.random.default_rng(seed)
    s = np.column_stack([rng.uniform(-1, 1, n), rng.laplace(0, 0.8, n), rng.normal(0, 0.3, n)])
    a = np.array([[0.9, 0.4, 0.1], [0.3, -0.8, 0.5], [0.2, 0.3, 0.9]])
    return s @ a.T, s


def test_recovers_components():
    x, s = _mixed()
    out = fast_ica(x, n_comp=3, seed=0)
    sh = np.asarray(out["sources"])
    # each true source has a high-|corr| match
    for j in range(2):  # uniform + laplace are the non-gaussian ones
        best = max(abs(float(np.corrcoef(s[:, j], sh[:, i])[0, 1])) for i in range(3))
        assert best > 0.9


def test_whitened_sources_uncorrelated():
    x, _ = _mixed()
    out = fast_ica(x, n_comp=3, seed=1)
    sh = np.asarray(out["sources"])
    c = np.corrcoef(sh, rowvar=False)
    off = c - np.eye(3)
    assert np.abs(off).max() < 0.15


def test_default_ncomp_equals_p():
    x, _ = _mixed()
    out = fast_ica(x, seed=0)
    assert out["n_comp"] == 3.0


def test_deterministic():
    x, _ = _mixed()
    a = fast_ica(x, n_comp=2, seed=3)
    b = fast_ica(x, n_comp=2, seed=3)
    assert np.allclose(a["w_ic"], b["w_ic"])


def test_fail_closed_1d():
    with pytest.raises(ValueError):
        fast_ica(np.ones(50))


def test_fail_closed_too_many_comp():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    with pytest.raises(ValueError):
        fast_ica(x, n_comp=4)


def test_bench():
    out = bench_fastica()
    assert out["synthetic_score"] == 1.0
