"""Tests for bandi_russell — noise/volatility separation."""

import numpy as np
import pytest

from quant_fund.models.bandi_russell import (
    bench_bandi_russell,
    br_realized_variance,
    noise_variance,
    optimal_m,
    synth_br,
)


def test_naive_rv_biased_br_better() -> None:
    d = synth_br(seed=1)
    r = br_realized_variance(np.asarray(d["r"]))
    iv = float(d["iv_true"])
    assert abs(r["rv_full"] - iv) / iv > 1.0
    assert abs(r["rv_br"] - iv) / iv < 0.35


def test_noise_variance_estimate() -> None:
    d = synth_br(seed=2)
    r = br_realized_variance(np.asarray(d["r"]))
    # eta2_hat = rv_full/(2T) ~ (IV + 2T eta^2)/(2T) = eta^2 + IV/(2T)
    assert r["eta2"] > float(d["eta2_true"])
    assert r["eta2"] < float(d["eta2_true"]) + r["rv_full"] / (2 * r["t"]) + 1e-9


def test_optimal_m_interior() -> None:
    m = optimal_m(4000, 5e-4, 1.0)
    assert 2 <= m <= 500


def test_no_noise_limit() -> None:
    d = synth_br(seed=3, eta=1e-6)
    r = br_realized_variance(np.asarray(d["r"]))
    assert abs(r["rv_br"] - float(d["iv_true"])) / float(d["iv_true"]) < 0.5


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        br_realized_variance(np.ones(10))
    with pytest.raises(ValueError):
        br_realized_variance(np.array([np.nan] * 100))
    with pytest.raises(ValueError):
        noise_variance(np.ones(5))
    with pytest.raises(ValueError):
        optimal_m(4000, 0.0, 1.0)
    with pytest.raises(ValueError):
        optimal_m(4000, 1e-3, -1.0)


def test_determinism() -> None:
    d = synth_br(seed=6)
    a = br_realized_variance(np.asarray(d["r"]))
    b = br_realized_variance(np.asarray(d["r"]))
    assert a == b


def test_bench_schema_and_score() -> None:
    r = bench_bandi_russell()
    for k in ("rv_br", "iv_true", "err_br", "err_naive", "improvement", "m_star", "score"):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
