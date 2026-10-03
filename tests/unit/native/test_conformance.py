"""Tests for native/conformance.py — float-contract enforcement."""

from __future__ import annotations

import numpy as np

from quant_fund.native.conformance import _compare, _invariants, run_conformance


def test_bit_exact_nan_placement():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, np.nan, 3.0])
    assert _compare("rolling_mean", a, b)[0]
    c = np.array([1.0, 2.0, 3.0])
    assert not _compare("rolling_mean", a, c)[0]  # NaN placement differs


def test_tolerant_allows_tiny_drift_not_nan_moved():
    a = np.array([1.0, np.nan])
    b = np.array([1.0 + 1e-13, np.nan])
    assert _compare("ema", a, b)[0]
    b2 = np.array([1.0 + 1e-13, 2.0])
    assert not _compare("ema", a, b2)[0]


def test_bit_exact_rejects_any_drift():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0 + 1e-15])
    assert not _compare("rolling_mean", a, b)[0]


def test_invariants_catch_broken_reference():
    assert _invariants("rolling_std", np.array([-1.0])) == ["rolling_std negative"]
    bad_bol = {"lower": np.array([2.0]), "mid": np.array([1.0]), "upper": np.array([3.0])}
    assert "bollinger mid outside band" in _invariants("bollinger", bad_bol)
    assert _invariants("hash_bytes", "xyz")


def test_run_conformance_structure():
    out = run_conformance(seed=1)
    assert out["schema"] == "native_conformance.v1"
    claim = out["claim"]
    assert claim["n_kernels"] == 12
    for v in claim["per_kernel"].values():
        assert v["verdict"] in {"bit_exact_ok", "within_tol", "mismatch", "reference_only"}
        assert v["n_cases"] > 0
    assert out["receipt_sha256"]


def test_determinism():
    a = run_conformance(seed=9)
    b = run_conformance(seed=9)
    assert a["receipt_sha256"] == b["receipt_sha256"]
