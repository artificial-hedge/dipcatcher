"""Tests for henze_zirkler — HZ multivariate normality."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.henze_zirkler import bench_henze_zirkler, henze_zirkler


def test_accepts_mvn():
    rng = np.random.default_rng(0)
    x = rng.multivariate_normal(np.zeros(3), np.eye(3) * 2, size=300)
    out = henze_zirkler(x)
    assert out["p"] > 0.01
    assert out["hz"] > 0


def test_rejects_heavy_tail():
    rng = np.random.default_rng(1)
    x = rng.standard_t(2.0, size=(300, 3)) * 2.0
    out = henze_zirkler(x)
    assert out["p"] < 0.05


def test_rejects_skewed():
    rng = np.random.default_rng(2)
    x = rng.lognormal(size=(400, 2))
    assert henze_zirkler(x)["p"] < 0.01


def test_statistic_scale():
    rng = np.random.default_rng(3)
    out = henze_zirkler(rng.standard_normal((200, 2)))
    assert out["mu_null"] > 0
    assert out["var_null"] > 0


def test_bad_inputs():
    with pytest.raises(ValueError):
        henze_zirkler(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        henze_zirkler(np.ones((5, 4)))


def test_bench():
    assert bench_henze_zirkler()["synthetic_score"] == 1.0
