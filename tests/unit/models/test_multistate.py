"""Adversarial probes for multistate."""

import numpy as np
import pytest

from quant_fund.models import multistate as ms


def test_intensity_rejects_out_of_range_dict_key():
    with pytest.raises(ValueError, match="out of range"):
        ms.ctmc_intensity_fit([(0, 1)], {5: 1.0}, n_states=3)


def test_intensity_rejects_noninteger_dict_key():
    with pytest.raises(ValueError, match="out of range"):
        ms.ctmc_intensity_fit([(0, 1)], {0.5: 1.0}, n_states=3)


def test_intensity_rejects_negative_exposure():
    with pytest.raises(ValueError, match="exposure"):
        ms.ctmc_intensity_fit([(0, 1)], {0: -1.0, 1: 2.0}, n_states=3)


def test_intensity_rejects_nan_exposure():
    with pytest.raises(ValueError, match="exposure"):
        ms.ctmc_intensity_fit([(0, 1)], {0: np.nan}, n_states=3)


def test_tp_rejects_negative_offdiagonal():
    bad = np.array([[-0.2, -0.05, 0.25], [0.0, -0.3, 0.3], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="off-diagonal"):
        ms.transition_probabilities(bad, 1.0)


def test_tp_rejects_nan_horizon():
    q = np.array([[-0.2, 0.15, 0.05], [0.0, -0.3, 0.3], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="horizon"):
        ms.transition_probabilities(q, np.nan)


def test_sojourn_rejects_invalid_generator():
    bad = np.array([[0.2, 0.15, 0.05], [0.0, -0.3, 0.3], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="generator"):
        ms.mean_sojourn(bad)


def test_bench_smoke():
    out = ms.bench_multistate()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_intensity_relerr"] < 0.35
