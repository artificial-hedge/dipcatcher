"""Tests for Diebold-Yilmaz connectedness (models/connectedness.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.connectedness import (
    bench_connectedness,
    connectedness,
    synth_var_spill,
)


def test_spillover_raises_tci():
    y = synth_var_spill(spill=0.6, seed=37)
    y0 = synth_var_spill(spill=0.0, seed=37)
    assert connectedness(y)["tci"] > connectedness(y0)["tci"] + 5.0


def test_transmitter_net_positive():
    y = synth_var_spill(spill=0.6, seed=37)
    out = connectedness(y)
    assert float(out["net_0"]) > 0.0


def test_independent_low_tci():
    y = synth_var_spill(spill=0.0, rho_resid=0.0, seed=37)
    out = connectedness(y)
    assert float(out["tci"]) < 10.0


def test_stationary_eig():
    y = synth_var_spill(spill=0.6, seed=37)
    out = connectedness(y)
    assert float(out["max_eig"]) < 1.0


def test_validation():
    y = synth_var_spill(seed=37)
    with pytest.raises(ValueError):
        connectedness(y[:50])  # T too small
    with pytest.raises(ValueError):
        connectedness(y[:, :1])  # k<2
    with pytest.raises(ValueError):
        connectedness(y, p=0)
    with pytest.raises(ValueError):
        connectedness(y, h=3)
    bad = y.copy()
    bad[10, 0] = np.nan
    with pytest.raises(ValueError):
        connectedness(bad)


def test_determinism():
    y = synth_var_spill(seed=37)
    a = connectedness(y)
    b = connectedness(y)
    assert float(a["tci"]) == float(b["tci"])


def test_bench_keys():
    out = bench_connectedness()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
