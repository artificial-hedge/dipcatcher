"""Tests for extremogram (wave-58)."""

import numpy as np
import pytest

from quant_fund.models.extremogram import (
    bench_extremogram,
    extremal_index,
    extremogram,
    synth_extremogram,
)


def test_clustered_extremogram_decays() -> None:
    x_ar, _ = synth_extremogram(seed=1)
    g = extremogram(x_ar, q=0.95, max_lag=8)
    assert g["extremogram"][0] > 0.4
    assert g["extremogram"][0] > g["extremogram"][4]


def test_extremal_index_separates() -> None:
    x_ar, x_iid = synth_extremogram(seed=2)
    assert extremal_index(x_ar)["theta"] < extremal_index(x_iid)["theta"]


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        extremogram(np.ones(50))
    with pytest.raises(ValueError):
        extremogram(np.full(500, np.nan))
    with pytest.raises(ValueError):
        extremal_index(np.ones(300))


def test_determinism() -> None:
    x, _ = synth_extremogram(seed=3)
    a = extremogram(x)
    b = extremogram(x)
    assert np.allclose(a["extremogram"], b["extremogram"])


def test_bench_schema_and_score() -> None:
    r = bench_extremogram()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
