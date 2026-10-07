"""Adversarial probes for markov_switching."""

import numpy as np
import pytest

from quant_fund.models import markov_switching as ms


def _y(n: int = 200):
    return np.asarray(ms.synth_markov(n=n, seed=0)["y"])


MU = np.array([-0.3, 0.8])
SIG = np.array([0.4, 1.2])
P = np.array([[0.95, 0.05], [0.05, 0.95]])


def test_filter_rejects_nan_mu():
    with pytest.raises(ValueError, match="finite"):
        ms.hamilton_filter(_y(), np.array([np.nan, 0.8]), SIG, P)


def test_filter_rejects_nan_sigma():
    with pytest.raises(ValueError, match="finite"):
        ms.hamilton_filter(_y(), MU, np.array([0.4, np.nan]), P)


def test_filter_rejects_nan_init():
    with pytest.raises(ValueError, match="initial"):
        ms.hamilton_filter(_y(), MU, SIG, P, init=np.array([np.nan, 1.0]))


def test_forecast_rejects_fractional_h():
    filt = ms.hamilton_filter(_y(), MU, SIG, P)
    with pytest.raises(ValueError, match="integer"):
        ms.regime_forecast(filt, h=2.5)


def test_synth_rejects_p_stay_out_of_range():
    with pytest.raises(ValueError, match="p_stay"):
        ms.synth_markov(n=100, p_stay=1.5)


def test_forecast_var_nonneg():
    filt = ms.hamilton_filter(_y(), MU, SIG, P)
    out = ms.regime_forecast(filt, h=10)
    assert float(np.asarray(out["var"])[0]) >= 0.0


def test_bench_smoke():
    out = ms.bench_markov_switching()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_regime_accuracy"] > 0.8
