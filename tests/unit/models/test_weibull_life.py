"""Tests for the Weibull life-data MLE (models/weibull_life.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models import weibull_life as wl


def test_weibull_fit_fails_closed_when_all_censored() -> None:
    """Zero observed failures => no interior likelihood maximum; must
    raise rather than return NaN/degenerate (beta, eta)."""
    t = np.array([10.0, 20.0, 30.0, 40.0])
    failed = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError, match="observed failure"):
        wl._weibull_fit(t, failed)


def test_weibull_bench_recovers_planted_params() -> None:
    out = wl.bench_weibull_life()
    assert out["synthetic_wb_beta_err"] < 0.5
    assert out["synthetic_wb_eta_rel"] < 0.1
