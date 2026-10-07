"""Tests for ross_recovery — Ross recovery of physical transitions."""

import numpy as np
import pytest

from quant_fund.models.ross_recovery import (
    bench_ross_recovery,
    ross_recover,
    synth_ross,
)


def test_exact_recovery_planted() -> None:
    d = synth_ross(seed=3)
    r = ross_recover(np.asarray(d["q"]))
    np.testing.assert_allclose(np.asarray(r["p"]), np.asarray(d["p_true"]), atol=1e-8)
    assert r["gamma"] == pytest.approx(float(d["gamma_true"]), abs=1e-8)


def test_recovered_is_row_stochastic() -> None:
    d = synth_ross(seed=9, n=4)
    r = ross_recover(np.asarray(d["q"]))
    p = np.asarray(r["p"])
    np.testing.assert_allclose(p.sum(axis=1), np.ones(4), atol=1e-8)
    assert np.all(p >= 0.0)


def test_u_recovers_direction() -> None:
    d = synth_ross(seed=5)
    r = ross_recover(np.asarray(d["q"]))
    np.testing.assert_allclose(np.asarray(r["u"]), np.asarray(d["u_true"]), atol=1e-6)


def test_fail_closed_inputs() -> None:
    with pytest.raises(ValueError):
        ross_recover(np.ones((3, 2)))
    q = np.ones((3, 3))
    q[1, 1] = -0.5
    with pytest.raises(ValueError):
        ross_recover(q)
    with pytest.raises(ValueError):
        ross_recover(np.full((3, 3), np.nan))
    with pytest.raises(ValueError):
        ross_recover(np.ones((1, 1)))


def test_determinism() -> None:
    d = synth_ross(seed=7)
    a = ross_recover(np.asarray(d["q"]))
    b = ross_recover(np.asarray(d["q"]))
    np.testing.assert_array_equal(a["p"], b["p"])
    assert a["gamma"] == b["gamma"]


def test_different_sizes() -> None:
    for n in (2, 8):
        d = synth_ross(seed=11, n=n)
        r = ross_recover(np.asarray(d["q"]))
        assert float(r["row_sum_err"]) < 1e-8


def test_bench_schema_and_score() -> None:
    r = bench_ross_recovery(seed=3)
    for k in (
        "synthetic_p_err",
        "synthetic_gamma_err",
        "synthetic_row_sum_err",
        "synthetic_gamma",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
