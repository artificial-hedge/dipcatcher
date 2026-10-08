"""Unit tests for quant_fund.models.chow_lin."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.chow_lin import (
    _agg_matrix,
    bench_chow_lin,
    chow_lin,
    denton_proportional,
    synth_disagg,
)


def test_annual_additivity() -> None:
    y_lo, ind, _ = synth_disagg(seed=1)
    out = chow_lin(y_lo, ind)
    yh = np.asarray(out["y_hat"])
    a = _agg_matrix(y_lo.size, yh.size // y_lo.size)
    assert np.max(np.abs(a @ yh - y_lo)) < 1e-5


def test_beats_uniform_spread() -> None:
    y_lo, ind, lat = synth_disagg(seed=2)
    out = chow_lin(y_lo, ind)
    yh = np.asarray(out["y_hat"])
    s = yh.size // y_lo.size
    naive = np.repeat(y_lo / s, s)
    assert np.sqrt(np.mean((yh - lat) ** 2)) < np.sqrt(np.mean((naive - lat) ** 2))


def test_denton_additivity_and_beats_naive() -> None:
    y_lo, ind, lat = synth_disagg(seed=3)
    out = denton_proportional(y_lo, ind)
    yh = np.asarray(out["y_hat"])
    s = yh.size // y_lo.size
    a = _agg_matrix(y_lo.size, s)
    assert np.max(np.abs(a @ yh - y_lo)) < 1e-5
    naive = np.repeat(y_lo / s, s)
    assert np.sqrt(np.mean((yh - lat) ** 2)) < np.sqrt(np.mean((naive - lat) ** 2))


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        chow_lin(np.ones(4), np.ones(16))
    y_lo, ind, _ = synth_disagg(seed=4)
    with pytest.raises(ValueError):
        chow_lin(y_lo, ind[:7])


def test_bench_contract() -> None:
    out = bench_chow_lin()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_cl_agg_err"] < 1e-5
