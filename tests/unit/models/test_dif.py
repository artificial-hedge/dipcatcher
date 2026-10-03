"""Tests for dif — differential item functioning."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dif import bench_dif, logistic_dif, mh_dif


def _sim(seed, dif=True, n=400):
    rng = np.random.default_rng(seed)
    score = rng.standard_normal(n)
    group = (rng.random(n) < 0.5).astype(float)
    logit = 1.2 * score - (1.0 if dif else 0.0) * group
    p = 1.0 / (1.0 + np.exp(-logit))
    return (rng.random(n) < p).astype(float), score, group


def test_dif_detected():
    y, s, g = _sim(0)
    out = mh_dif(y, s, g)
    assert out["p"] < 0.05
    assert out["delta_mh"] > 1.0  # focal penalized -> positive delta (ref favored)


def test_no_dif_clean():
    y, s, g = _sim(1, dif=False)
    out = mh_dif(y, s, g)
    assert out["p"] > 0.01


def test_logistic_uniform_dif():
    y, s, g = _sim(2)
    out = logistic_dif(y, s, g)
    assert out["p_uniform"] < 0.05


def test_fail_closed_nonbinary():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        mh_dif(rng.random(50), rng.standard_normal(50), (rng.random(50) < 0.5).astype(float))


def test_bench():
    out = bench_dif()
    assert out["score"] == 1.0
