"""Tests for engle_granger — two-step cointegration."""

import numpy as np
import pytest

from quant_fund.models.engle_granger import (
    bench_engle_granger,
    ecm_fit,
    eg_tau,
    synth_eg,
)


def test_cointegrated_pair_rejected() -> None:
    y1, x1, _, _ = synth_eg(seed=1)
    r = eg_tau(y1, x1)
    assert r["tau"] < r["crit5"]


def test_independent_pair_passes() -> None:
    _, _, y0, x0 = synth_eg(seed=2)
    r = eg_tau(y0, x0)
    assert r["tau"] > r["crit5"]


def test_ecm_negative_adjustment() -> None:
    y1, x1, _, _ = synth_eg(seed=3)
    r = ecm_fit(y1, x1)
    assert r["alpha"] < 0
    assert r["alpha_p"] < 0.05


def test_beta_recovers_loading() -> None:
    y1, x1, _, _ = synth_eg(seed=4)
    r = eg_tau(y1, x1)
    assert 0.5 < r["beta"] < 1.5


def test_fail_closed() -> None:
    y1, x1, _, _ = synth_eg(seed=5)
    with pytest.raises(ValueError):
        eg_tau(y1[:30], x1[:30])
    with pytest.raises(ValueError):
        eg_tau(np.full(100, np.nan), x1[:100])
    with pytest.raises(ValueError):
        eg_tau(np.ones(100), x1[:100])
    with pytest.raises(ValueError):
        ecm_fit(y1[:30], x1[:30])


def test_determinism() -> None:
    y1, x1, _, _ = synth_eg(seed=6)
    assert eg_tau(y1, x1) == eg_tau(y1, x1)


def test_bench_schema_and_score() -> None:
    r = bench_engle_granger()
    for k in (
        "synthetic_tau_ci",
        "synthetic_crit5",
        "synthetic_tau_nc",
        "synthetic_alpha",
        "synthetic_alpha_p",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0
