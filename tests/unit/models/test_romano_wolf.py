"""Tests for romano_wolf — stepwise FWER multiple testing."""

import numpy as np
import pytest

from quant_fund.models.romano_wolf import bench_romano_wolf, rw_stepdown, synth_rw


def test_strong_strategies_rejected() -> None:
    x = synth_rw(seed=1)
    r = rw_stepdown(x, alpha=0.10, n_boot=150, seed=1)
    assert {0, 1}.issubset(set(r["rejects"]))


def test_null_strategies_kept() -> None:
    x = synth_rw(seed=2)
    r = rw_stepdown(x, alpha=0.10, n_boot=150, seed=2)
    assert set(r["rejects"]).isdisjoint({3, 4, 5})


def test_padj_monotone_in_order() -> None:
    x = synth_rw(seed=3)
    r = rw_stepdown(x, alpha=0.10, n_boot=150, seed=3)
    padj = np.asarray(r["p_adj"])
    t = np.asarray(r["t_stats"])
    order = np.argsort(-t)
    assert np.all(np.diff(padj[order]) >= -1e-9)


def test_all_null_rejects_nothing_mostly() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((500, 4))
    r = rw_stepdown(x, alpha=0.05, n_boot=150, seed=4)
    assert len(list(r["rejects"])) <= 1


def test_fail_closed() -> None:
    x = synth_rw(seed=5)
    with pytest.raises(ValueError):
        rw_stepdown(x[:40])
    with pytest.raises(ValueError):
        rw_stepdown(x[:, :1])
    with pytest.raises(ValueError):
        rw_stepdown(np.full((300, 4), np.nan))


def test_determinism() -> None:
    x = synth_rw(seed=6)
    a = rw_stepdown(x, n_boot=100, seed=6)
    b = rw_stepdown(x, n_boot=100, seed=6)
    assert a["rejects"] == b["rejects"]
    np.testing.assert_array_equal(a["p_adj"], b["p_adj"])


def test_bench_schema_and_score() -> None:
    r = bench_romano_wolf()
    for k in (
        "synthetic_n_reject",
        "synthetic_n_bonf_reject",
        "synthetic_strongest_survives",
        "synthetic_null_dropped",
        "synthetic_min_null_padj",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
