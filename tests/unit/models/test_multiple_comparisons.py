"""Tests for multiple_comparisons — Tukey/Dunnett/GH/Scheffe."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.multiple_comparisons import (
    bench_multiple_comparisons,
    dunnett_test,
    games_howell,
    scheffe_test,
    tukey_hsd,
)


def _three_groups(shifted: bool = True, seed: int = 0, n: int = 40):
    rng = np.random.default_rng(seed)
    ys, gs = [], []
    mus = [0.0, 0.0, 1.8] if shifted else [0.0, 0.0, 0.0]
    for j, mu in enumerate(mus):
        ys.append(rng.normal(loc=mu, size=n))
        gs.append(np.full(n, j))
    return np.concatenate(ys), np.concatenate(gs)


def test_tukey_flags_shift():
    y, g = _three_groups()
    out = tukey_hsd(y, g)
    p = np.asarray(out["pair_p"])
    assert p[0, 2] < 0.01 and p[1, 2] < 0.01
    assert p[0, 1] > 0.1


def test_tukey_clean_when_unshifted():
    y, g = _three_groups(shifted=False)
    out = tukey_hsd(y, g)
    p = np.asarray(out["pair_p"])
    assert np.nanmin(p) > 0.2


def test_dunnett_vs_control():
    y, g = _three_groups()
    out = dunnett_test(y, g, control=0, seed=1, mc=1500)
    p = np.asarray(out["pair_p"])
    # p[0] = group1-vs-control (no shift), p[1] = group2-vs-control (shifted)
    assert p[0] > 0.1 and p[1] < 0.05


def test_games_howell_runs_unequal_var():
    rng = np.random.default_rng(2)
    a = rng.normal(size=40)
    b = rng.normal(scale=3.0, size=40) + 1.0
    c = rng.normal(scale=0.5, size=40)
    y = np.concatenate([a, b, c])
    g = np.concatenate([np.zeros(40), np.ones(40), np.full(40, 2)])
    out = games_howell(y, g)
    p = np.asarray(out["pair_p"])
    assert np.isfinite(np.nanmin(p))


def test_scheffe_conservative():
    y, g = _three_groups()
    st = scheffe_test(y, g)
    tk = tukey_hsd(y, g)
    ps = np.asarray(st["pair_p"])
    pt = np.asarray(tk["pair_p"])
    # scheffe p >= tukey p on the same layout (conservative)
    assert np.all(np.nan_to_num(ps, nan=1.0) >= np.nan_to_num(pt, nan=1.0) - 1e-9)


def test_fail_closed_one_group():
    with pytest.raises(ValueError):
        tukey_hsd(np.ones(10), np.zeros(10))


def test_bench():
    out = bench_multiple_comparisons()
    assert out["synthetic_score"] == 1.0
