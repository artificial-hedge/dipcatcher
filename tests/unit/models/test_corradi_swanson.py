"""Tests for corradi_swanson — OOS predictive accuracy."""

import numpy as np
import pytest

from quant_fund.models.corradi_swanson import (
    bench_corradi_swanson,
    cs_test,
    synth_cs,
)


def test_rejects_under_alternative() -> None:
    y = synth_cs(seed=1)
    r = cs_test(y, n_boot=200, seed=1)
    assert r["mspe_diff"] > 0.0
    assert r["p_boot"] < 0.1


def test_no_reject_under_null() -> None:
    y = synth_cs(seed=2, under_null=True)
    r = cs_test(y, n_boot=200, seed=2)
    assert r["p_boot"] > 0.05


def test_mspe_small_ge_big_when_ar_true() -> None:
    y = synth_cs(seed=3)
    r = cs_test(y, n_boot=100, seed=3)
    assert r["mspe_big"] < r["mspe_small"]


def test_enc_new_positive_under_alternative() -> None:
    y = synth_cs(seed=4)
    r = cs_test(y, n_boot=100, seed=4)
    assert r["enc_new"] > 0.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        cs_test(np.ones(50))
    with pytest.raises(ValueError):
        cs_test(np.array([np.nan] * 200))


def test_determinism() -> None:
    y = synth_cs(seed=6)
    a = cs_test(y, n_boot=100, seed=6)
    b = cs_test(y, n_boot=100, seed=6)
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_corradi_swanson()
    for k in (
        "synthetic_mspe_diff_alt",
        "synthetic_p_alt",
        "synthetic_p_null",
        "synthetic_enc_new_alt",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
