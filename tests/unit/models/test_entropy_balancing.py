"""Tests for entropy balancing (models/entropy_balancing.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.entropy_balancing import (
    att_entropy,
    balance_table,
    bench_entropy_balancing,
    entropy_weights,
    synth_eb,
)


def _panel(**kw):
    return synth_eb(seed=12, **kw)


def test_exact_balance_when_feasible():
    d = _panel()
    x = np.asarray(d["x"])
    dd = np.asarray(d["d"]).astype(bool)
    out = entropy_weights(x[dd], x[~dd])
    assert out["converged"] == 1.0
    assert out["max_gap_raw"] < 1e-3


def test_weights_positive_and_normalized():
    d = _panel()
    x = np.asarray(d["x"])
    dd = np.asarray(d["d"]).astype(bool)
    w = np.asarray(entropy_weights(x[dd], x[~dd])["weights"])
    assert np.all(w > 0)
    n_c = int((~dd).sum())
    assert abs(w.sum() - n_c) < 1e-6


def test_att_recovery():
    d = _panel(effect=1.2)
    est = att_entropy(
        np.asarray(d["y"]).ravel(),
        np.asarray(d["d"]),
        np.asarray(d["x"]),
    )
    assert abs(est["att"] - 1.2) < 0.5


def test_null_att_small():
    d = _panel(effect=0.0)
    est = att_entropy(
        np.asarray(d["y"]).ravel(),
        np.asarray(d["d"]),
        np.asarray(d["x"]),
    )
    assert abs(est["att"]) < 0.5


def test_balance_table_reduces_smd():
    d = _panel()
    x = np.asarray(d["x"])
    dd = np.asarray(d["d"]).astype(bool)
    w = np.asarray(entropy_weights(x[dd], x[~dd])["weights"])
    bal = balance_table(x, dd, w)
    assert bal["max_smd_w"] < bal["max_smd"]


def test_ess_reasonable():
    d = _panel()
    x = np.asarray(d["x"])
    dd = np.asarray(d["d"]).astype(bool)
    out = entropy_weights(x[dd], x[~dd])
    assert out["ess"] > 20.0


def test_validation():
    d = _panel()
    x = np.asarray(d["x"])
    dd = np.asarray(d["d"]).astype(bool)
    with pytest.raises(ValueError):
        entropy_weights(x[dd, :2], x[~dd])
    with pytest.raises(ValueError):
        entropy_weights(np.ones((1, 3)), x[~dd])
    with pytest.raises(ValueError):
        entropy_weights(x[dd], x[~dd][:2])
    with pytest.raises(ValueError):
        att_entropy(np.ones(4), np.ones(4), np.ones((4, 2)))
    with pytest.raises(ValueError):
        balance_table(x, np.ones(x.shape[0]))


def test_determinism():
    d = _panel()
    x = np.asarray(d["x"])
    dd = np.asarray(d["d"]).astype(bool)
    a = entropy_weights(x[dd], x[~dd])
    b = entropy_weights(x[dd], x[~dd])
    assert np.array_equal(np.asarray(a["weights"]), np.asarray(b["weights"]))


def test_bench_keys():
    out = bench_entropy_balancing()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_converged"] == 1.0
    assert out["synthetic_beats_raw"] == 1.0
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
