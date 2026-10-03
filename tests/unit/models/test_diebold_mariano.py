"""Tests for diebold_mariano — DM/HLN predictive-accuracy tests."""

import numpy as np
import pytest

from quant_fund.models.diebold_mariano import (
    bench_diebold_mariano,
    dm_test,
    synth_dm,
)


def test_rejects_worse_forecaster() -> None:
    e_bad, e_good, _ = synth_dm(seed=1)
    r = dm_test(e_bad, e_good)
    assert r["p_hln"] < 0.01
    assert r["mean_diff"] > 0


def test_accepts_equal_accuracy() -> None:
    _, e_good, e_same = synth_dm(seed=2)
    r = dm_test(e_good, e_same)
    assert r["p_hln"] > 0.9


def test_hln_correction_direction() -> None:
    e_bad, e_good, _ = synth_dm(seed=3)
    r = dm_test(e_bad, e_good, h=1)
    assert abs(r["dm_hln"]) < abs(r["dm"])  # finite-sample shrink


def test_absolute_loss_variant() -> None:
    e_bad, e_good, _ = synth_dm(seed=4)
    r = dm_test(e_bad, e_good, loss="ae")
    assert np.isfinite(r["dm_hln"])


def test_fail_closed() -> None:
    e_bad, e_good, _ = synth_dm(seed=5)
    with pytest.raises(ValueError):
        dm_test(e_bad[:20], e_good[:20])
    with pytest.raises(ValueError):
        dm_test(np.full(100, np.nan), e_good[:100])
    with pytest.raises(ValueError):
        dm_test(e_bad, e_good, h=0)
    with pytest.raises(ValueError):
        dm_test(e_bad[:50], e_good[:60])


def test_determinism() -> None:
    e_bad, e_good, _ = synth_dm(seed=6)
    assert dm_test(e_bad, e_good) == dm_test(e_bad, e_good)


def test_bench_schema_and_score() -> None:
    r = bench_diebold_mariano()
    for k in ("dm_stat", "dm_hln", "p_alt", "p_null", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
