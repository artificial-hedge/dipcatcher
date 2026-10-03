"""Unit tests for quant_fund.models.fire_sales."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fire_sales import (
    bench_fire_sales,
    fire_sales,
    synth_fire,
)


def test_overcap_book_cascades() -> None:
    d = synth_fire(seed=1)
    out = fire_sales(d["holdings"], d["leverage"], d["depth"], lev_cap=3.0)
    assert out["total_sales"] > 0.0
    assert out["price_drop_max"] > 0.05
    assert out["rounds"] >= 1


def test_slack_book_quiet() -> None:
    d = synth_fire(seed=1)
    out = fire_sales(d["holdings"], d["leverage"], d["depth"], lev_cap=6.0)
    assert out["total_sales"] == 0.0
    assert out["price_drop_max"] == 0.0
    assert out["rounds"] == 0.0


def test_thin_asset_drops_most() -> None:
    d = synth_fire(seed=2)
    out = fire_sales(d["holdings"], d["leverage"], d["depth"], lev_cap=3.0)
    # asset 0 is the thin shared one
    assert out["price_drop_max"] > 0.3


def test_determinism() -> None:
    d = synth_fire(seed=5)
    a = fire_sales(d["holdings"], d["leverage"], d["depth"], lev_cap=3.0)
    b = fire_sales(d["holdings"], d["leverage"], d["depth"], lev_cap=3.0)
    assert a == b


def test_validation() -> None:
    d = synth_fire(seed=1)
    with pytest.raises(ValueError):
        fire_sales(np.ones((3, 2)), np.ones(3), np.zeros(2))  # zero depth
    with pytest.raises(ValueError):
        fire_sales(np.ones((3, 2)), np.ones(2), np.ones(2))  # leverage mismatch
    with pytest.raises(ValueError):
        fire_sales(np.ones(3), np.ones(3), np.ones(3))  # not a matrix
    with pytest.raises(ValueError):
        fire_sales(d["holdings"], d["leverage"], d["depth"], lev_cap=-1.0)
    with pytest.raises(ValueError):
        fire_sales(np.full((2, 2), np.nan), np.ones(2), np.ones(2))


def test_schema() -> None:
    d = synth_fire(seed=1)
    out = fire_sales(d["holdings"], d["leverage"], d["depth"])
    assert set(out) == {
        "total_sales",
        "price_drop_max",
        "rounds",
        "n_defaulted",
        "book_loss_frac",
        "equity_loss_frac",
    }


def test_bench_keys_and_pass() -> None:
    out = bench_fire_sales()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_total_sales",
        "synthetic_price_drop",
        "synthetic_rounds",
        "synthetic_quiet_sales",
        "synthetic_n_defaulted",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
