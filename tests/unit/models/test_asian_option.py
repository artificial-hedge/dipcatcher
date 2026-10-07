"""Unit tests for quant_fund.models.asian_option."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.asian_option import (
    arith_asian_mc,
    geo_asian_call,
    synth_asian,
    tw_asian_call,
)


def test_geo_closed_form_pos() -> None:
    assert geo_asian_call(100.0, 100.0, 1.0, 0.03, 0.25) > 0.0


def test_arith_exceeds_geo() -> None:
    r = arith_asian_mc(100.0, 100.0, 1.0, 0.03, 0.25, n_paths=4000, seed=1)
    assert r["price"] > r["geo_price"]


def test_tw_close_to_mc() -> None:
    r = arith_asian_mc(100.0, 100.0, 1.0, 0.03, 0.25, n_paths=4000, seed=2)
    tw = tw_asian_call(100.0, 100.0, 1.0, 0.03, 0.25)
    assert abs(r["price"] - tw) < 0.15 * tw


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        geo_asian_call(0.0, 100.0, 1.0, 0.03, 0.25)
    with pytest.raises(ValueError):
        arith_asian_mc(100.0, -5.0, 1.0, 0.03, 0.25)


def test_bench_contract() -> None:
    out = synth_asian()
    assert out["score"] == 1.0
    assert out["synthetic_ao_mc"] > out["synthetic_ao_geo"]


def test_tw_asian_call_zero_rate_finite() -> None:
    # r = 0 is a removable singularity, not a NaN: (e^{rT}-1)/(rT) -> 1.
    v = tw_asian_call(100.0, 100.0, 1.0, 0.0, 0.25)
    assert np.isfinite(v) and v > 0.0
    # Continuity against a tiny nonzero rate.
    v_eps = tw_asian_call(100.0, 100.0, 1.0, 1e-9, 0.25)
    assert v == pytest.approx(v_eps, rel=1e-6)
    # Jensen: arithmetic-mean price still exceeds the geometric one.
    assert v > geo_asian_call(100.0, 100.0, 1.0, 0.0, 0.25)


def test_tw_asian_call_small_rate_stable() -> None:
    # expm1 form keeps small-but-nonzero rates accurate (no 0/0-style loss).
    v = tw_asian_call(80.0, 90.0, 0.5, 1e-8, 0.4)
    assert np.isfinite(v) and v >= 0.0
