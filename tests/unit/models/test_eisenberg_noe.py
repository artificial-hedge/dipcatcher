"""Unit tests for quant_fund.models.eisenberg_noe."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.eisenberg_noe import (
    bench_eisenberg_noe,
    eisenberg_noe,
    synth_clearing,
)

_KEYS = {
    "payments_sum",
    "obligations_sum",
    "shortfall",
    "n_defaults",
    "n_negative_worth",
    "min_worth",
    "iterations",
    "_payments",
}


def test_detects_shock_cascade() -> None:
    d = synth_clearing(seed=1)
    out = eisenberg_noe(d["liabilities"], d["external_assets"])
    assert out["n_defaults"] >= 1
    assert out["shortfall"] > 0


def test_quiet_system_clears() -> None:
    d = synth_clearing(seed=1)
    out = eisenberg_noe(d["liabilities"], np.full(8, 16.0))
    assert out["n_defaults"] == 0
    assert out["shortfall"] < 1e-9
    assert out["payments_sum"] == pytest.approx(out["obligations_sum"], abs=1e-8)


def test_payments_never_exceed_obligations() -> None:
    d = synth_clearing(seed=7)
    out = eisenberg_noe(d["liabilities"], d["external_assets"])
    pays = np.asarray(out["_payments"])
    pbar = np.asarray(d["liabilities"]).sum(axis=1)
    assert np.all(pays <= pbar + 1e-9)
    assert np.all(pays >= -1e-12)


def test_schema() -> None:
    d = synth_clearing(seed=3)
    out = eisenberg_noe(d["liabilities"], d["external_assets"])
    assert set(out) == _KEYS


def test_determinism() -> None:
    d = synth_clearing(seed=5)
    a = eisenberg_noe(d["liabilities"], d["external_assets"])
    b = eisenberg_noe(d["liabilities"], d["external_assets"])
    assert np.array_equal(np.asarray(a["_payments"]), np.asarray(b["_payments"]))
    assert {k: v for k, v in a.items() if k != "_payments"} == {
        k: v for k, v in b.items() if k != "_payments"
    }


def test_validation() -> None:
    with pytest.raises(ValueError):
        eisenberg_noe(np.ones((3, 3)), np.ones(3))  # diagonal liabilities
    with pytest.raises(ValueError):
        eisenberg_noe(np.zeros((2, 3)), np.ones(2))  # not square
    with pytest.raises(ValueError):
        eisenberg_noe(np.eye(2) - np.eye(2), np.ones(3))  # shape mismatch
    with pytest.raises(ValueError):
        eisenberg_noe(np.zeros((2, 2)), np.array([-1.0, 0.0]))  # negative assets
    with pytest.raises(ValueError):
        eisenberg_noe(np.full((2, 2), np.nan), np.ones(2))  # non-finite


def test_bench_keys_and_pass() -> None:
    out = bench_eisenberg_noe()
    assert set(out) == {
        "synthetic_detects",
        "synthetic_determinism",
        "synthetic_n_defaults",
        "synthetic_shortfall",
        "synthetic_quiet_defaults",
        "synthetic_min_worth",
        "synthetic_iterations",
    }
    assert all(np.isfinite(v) for v in out.values())
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
