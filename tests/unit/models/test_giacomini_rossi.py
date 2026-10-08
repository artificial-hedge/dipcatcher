"""Tests for giacomini_rossi — forecast-comparison fluctuation test."""

import numpy as np
import pytest

from quant_fund.models.giacomini_rossi import (
    bench_giacomini_rossi,
    gr_fluctuation,
    synth_gr,
)


def test_accepts_stable() -> None:
    d = synth_gr(seed=1)
    r = gr_fluctuation(d, m=100, n_boot=150, seed=1)
    assert r["p_boot"] > 0.1


def test_rejects_breakdown() -> None:
    d = synth_gr(seed=2, break_frac=0.5)
    r = gr_fluctuation(d, m=100, n_boot=150, seed=2)
    assert r["p_boot"] < 0.05
    assert r["max_stat"] > r["crit_95"]


def test_break_location_detected() -> None:
    d = synth_gr(seed=3, break_frac=0.5)
    r = gr_fluctuation(d, m=100, n_boot=150, seed=3)
    assert abs(r["break_idx"] - 300) < 120


def test_tau_path_shape() -> None:
    d = synth_gr(seed=4)
    r = gr_fluctuation(d, m=100, n_boot=50, seed=4)
    tau = np.asarray(r["tau"])
    assert tau.size == d.size - 100 + 1
    assert np.all(np.isfinite(tau))


def test_fail_closed() -> None:
    d = synth_gr(seed=5)
    with pytest.raises(ValueError):
        gr_fluctuation(d[:50], m=20)
    with pytest.raises(ValueError):
        gr_fluctuation(d, m=10)
    with pytest.raises(ValueError):
        gr_fluctuation(np.zeros(400), m=100)  # zero variance
    with pytest.raises(ValueError):
        gr_fluctuation(np.full(400, np.nan), m=100)


def test_determinism() -> None:
    d = synth_gr(seed=6)
    a = gr_fluctuation(d, m=100, n_boot=50, seed=6)
    b = gr_fluctuation(d, m=100, n_boot=50, seed=6)
    np.testing.assert_array_equal(a["tau"], b["tau"])
    assert a["p_boot"] == b["p_boot"]


def test_bench_schema_and_score() -> None:
    r = bench_giacomini_rossi()
    for k in (
        "synthetic_p_null",
        "synthetic_p_alt",
        "synthetic_break_idx",
        "synthetic_max_stat_alt",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
