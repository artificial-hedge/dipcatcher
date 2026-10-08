"""Unit tests for quant_fund.models.stable_dist."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.stable_dist import (
    _mcculloch,
    bench_stable,
    stable_fit,
    synth_stable,
)


def test_alpha_recovery() -> None:
    x, _ = synth_stable(seed=1)
    assert abs(stable_fit(x)["alpha"] - 1.6) < 0.45


def test_mcculloch_neighborhood() -> None:
    x, _ = synth_stable(seed=2)
    a, b, g, d = _mcculloch(x)
    assert abs(a - 1.6) < 0.55 and g > 0


def test_heavier_tail_lower_alpha() -> None:
    x, x2 = synth_stable(seed=3)
    assert stable_fit(x2)["alpha"] < stable_fit(x)["alpha"]


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        stable_fit(np.ones(50))
    with pytest.raises(ValueError):
        stable_fit(np.full(200, 1.0))


def test_bench_contract() -> None:
    out = bench_stable()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_st_cf_err"] < 0.5
