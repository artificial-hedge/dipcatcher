"""Tests for rasch_fit — infit/outfit."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.rasch_fit import bench_rasch_fit, item_fit, person_fit


def _sim(seed=0, corrupt=False, n=200, k=10):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n)
    b = np.linspace(-1.5, 1.5, k)
    p = 1.0 / (1.0 + np.exp(-(theta[:, None] - b[None, :])))
    a = (rng.random((n, k)) < p).astype(float)
    if corrupt:
        a[:, 0] = (rng.random(n) < 0.5).astype(float)
    return a


def test_clean_items_near_one():
    a = _sim(0)
    out = item_fit(a)
    outfits = np.asarray(out["outfit"])
    assert 0.6 < outfits.mean() < 1.4


def test_corrupt_item_high_outfit():
    a = _sim(1, corrupt=True)
    outfits = np.asarray(item_fit(a)["outfit"])
    assert outfits[0] > outfits[1:].mean()


def test_person_fit_runs():
    a = _sim(2)
    out = person_fit(a)
    assert np.asarray(out["outfit"]).shape == (200,)


def test_fail_closed_nonbinary():
    with pytest.raises(ValueError):
        item_fit(np.random.default_rng(0).random((30, 6)))


def test_bench():
    out = bench_rasch_fit()
    assert out["score"] == 1.0
