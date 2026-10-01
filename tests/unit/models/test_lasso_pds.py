"""Tests for lasso_pds (wave-57)."""

import numpy as np
import pytest

from quant_fund.models.lasso_pds import (
    bench_lasso_pds,
    post_double_selection,
    synth_pds,
)


def test_ci_covers_true_effect() -> None:
    y, d, x, _ = synth_pds(seed=1)
    r = post_double_selection(y, d, x)
    assert r["ci_lo"] <= 1.0 <= r["ci_hi"]


def test_support_recovery() -> None:
    y, d, x, _ = synth_pds(seed=2)
    r = post_double_selection(y, d, x)
    assert r["n_sel"] >= 3.0


def test_fail_closed() -> None:
    y, d, x, _ = synth_pds(seed=3)
    with pytest.raises(ValueError):
        post_double_selection(np.full(150, np.nan), d, x)
    with pytest.raises(ValueError):
        post_double_selection(y, np.ones(150), x)
    with pytest.raises(ValueError):
        post_double_selection(y[:50], d, x)


def test_determinism() -> None:
    y, d, x, _ = synth_pds(seed=4)
    assert post_double_selection(y, d, x) == post_double_selection(y, d, x)


def test_bench_schema_and_score() -> None:
    r = bench_lasso_pds()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
