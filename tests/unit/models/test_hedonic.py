"""Unit tests for quant_fund.models.hedonic."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hedonic import (
    bench_hedonic,
    hedonic_index,
    repeat_sales_index,
    synth_hedonic,
)


def test_repeat_sales_tracks_true_index() -> None:
    d = synth_hedonic(seed=1)
    out = repeat_sales_index(d["price1"], d["price2"], d["period1"], d["period2"])
    true_g = float(np.asarray(d["true_index"])[-1])
    assert abs(out["index_last"] - true_g) / true_g < 0.15


def test_index_base_is_one() -> None:
    d = synth_hedonic(seed=2)
    out = repeat_sales_index(d["price1"], d["price2"], d["period1"], d["period2"])
    assert out["index_first"] == pytest.approx(1.0)


def test_hedonic_schema() -> None:
    rng = np.random.default_rng(0)
    n = 200
    periods = rng.integers(0, 5, n).astype(np.float64)
    chars = np.column_stack([rng.uniform(size=n), rng.uniform(size=n)])
    lp = 0.1 * periods + 0.3 * chars[:, 0] + rng.normal(0.0, 0.05, n)
    out = hedonic_index(lp, periods, chars)
    assert set(out) == {
        "index_first",
        "index_last",
        "index_growth",
        "attr_beta_mean",
        "r2",
        "n",
        "n_periods",
        "_index",
        "_periods",
    }
    assert out["index_growth"] > 1.0


def test_determinism() -> None:
    d = synth_hedonic(seed=5)
    a = repeat_sales_index(d["price1"], d["price2"], d["period1"], d["period2"])
    b = repeat_sales_index(d["price1"], d["price2"], d["period1"], d["period2"])
    assert np.array_equal(np.asarray(a["_index"]), np.asarray(b["_index"]))
    assert {k: v for k, v in a.items() if k not in {"_index", "_periods"}} == {
        k: v for k, v in b.items() if k not in {"_index", "_periods"}
    }


def test_validation() -> None:
    d = synth_hedonic(seed=1)
    with pytest.raises(ValueError):
        repeat_sales_index(np.ones(10), np.ones(10), np.zeros(10), np.ones(10))
    with pytest.raises(ValueError):
        repeat_sales_index(-d["price1"], d["price2"], d["period1"], d["period2"])
    with pytest.raises(ValueError):
        repeat_sales_index(d["price1"], d["price2"], d["period2"], d["period1"])  # t <= s
    with pytest.raises(ValueError):
        hedonic_index(np.ones(10), np.zeros(10), np.ones((10, 1)))


def test_bench_keys_and_pass() -> None:
    out = bench_hedonic()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_est_growth",
        "synthetic_true_growth",
        "synthetic_resid_sd",
        "synthetic_r2",
        "synthetic_n_pairs",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
