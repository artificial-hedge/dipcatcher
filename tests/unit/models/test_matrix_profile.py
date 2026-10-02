"""Tests for matrix profile, motif/discord and SAX."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.matrix_profile import (
    bench_matrix_profile,
    discord,
    matrix_profile,
    motif,
    sax,
)


def test_repeated_subsequence_motif():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300) * 0.1
    pat = np.sin(np.arange(20) / 3.0)
    x[40:60] += pat
    x[180:200] += pat
    mp = matrix_profile(x, m=20)
    i, j, d = motif(mp)
    assert d < 1.0
    assert {i, j} == {40, 180} or min(abs(i - 40), abs(j - 40)) <= 20


def test_discord_spike():
    x = np.sin(np.arange(200) / 10.0)
    x[100] += 8.0
    mp = matrix_profile(x, m=15)
    i, d = discord(mp)
    assert abs(i - 100) <= 15
    assert np.isfinite(d)


def test_profile_shape_and_exclusion():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(150)
    m = 12
    mp = matrix_profile(x, m=m)
    assert mp["profile"].shape == (150 - m + 1,)
    assert (np.asarray(mp["profile"]) >= 0).all()


def test_sax_symbols():
    x = np.sin(np.arange(100) / 5.0)
    w = sax(x, n_segments=10, alphabet=4)
    assert w.shape == (10,)
    assert w.min() >= 0 and w.max() <= 3


def test_fail_closed():
    with pytest.raises(ValueError):
        matrix_profile(np.arange(10.0), m=6)
    with pytest.raises(ValueError):
        matrix_profile(np.array([np.nan] * 50), m=5)
    with pytest.raises(ValueError):
        sax(np.arange(10.0), n_segments=20)


def test_bench():
    res = bench_matrix_profile()
    assert res["synthetic_score"] == 1.0
    assert res["synthetic_motif_found"] == 1.0
    assert res["synthetic_discord_found"] == 1.0
