"""Tests for the exact fractional Brownian sampler (Davies-Harte)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.models.fbm import (
    FBM_SCHEMA,
    _fgn_autocov,
    fbm_bench,
    fbm_circulant,
    fbm_contract_errors,
    fgn_circulant,
    write_fbm_receipt,
)
from quant_fund.research.lane_contracts import lane_contract_errors
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_fgn_deterministic() -> None:
    a = fgn_circulant(64, 0.1, 42, 8)
    b = fgn_circulant(64, 0.1, 42, 8)
    np.testing.assert_array_equal(a, b)


def test_fgn_variance_and_autocorr() -> None:
    x = fgn_circulant(256, 0.3, 5, 3000)
    assert abs(x.var() - 1.0) < 0.05
    for lag in (1, 2, 3):
        meas = float(np.mean(x[:, :-lag] * x[:, lag:]) / x.var())
        assert abs(meas - _fgn_autocov(lag, 0.3)) < 0.04


def test_brownian_limit_uncorrelated() -> None:
    x = fgn_circulant(256, 0.5, 5, 3000)
    assert abs(x.var() - 1.0) < 0.05
    for lag in (1, 2):
        meas = float(np.mean(x[:, :-lag] * x[:, lag:]) / x.var())
        assert abs(meas) < 0.05


def test_fbm_variance_scaling() -> None:
    h = 0.15
    b = fbm_circulant(512, h, 1.0 / 512, 3, 1500)
    assert b.shape == (1500, 513)
    var_b = b.var(axis=0)
    i0 = 128
    ratio = var_b[2 * i0] / var_b[i0]
    assert abs(ratio / 2.0 ** (2 * h) - 1.0) < 0.08


def test_fbm_bench_passes() -> None:
    payload = fbm_bench(n=128, n_paths=1500, seed=11)
    assert payload["schema"] == FBM_SCHEMA
    assert payload["data_label"] == "SYNTHETIC"
    c = payload["claim"]
    assert c["ok"]
    assert c["n_passed"] == c["n_probes"] == 5
    assert fbm_contract_errors(payload) == []


def test_fbm_contract_catches_forgery() -> None:
    payload = fbm_bench(n=128, n_paths=1500, seed=11)
    bad = json.loads(json.dumps(payload))
    bad["claim"]["n_passed"] = 0
    assert "n_passed_mismatch" in fbm_contract_errors(bad)
    bad2 = json.loads(json.dumps(payload))
    bad2["claim"]["results"]["hybrid_deficit_pinned"] = False
    assert "ok_mismatch" in fbm_contract_errors(bad2)
    bad3 = json.loads(json.dumps(payload))
    bad3["claim"]["hybrid_variance_ratio"] = {"median": 0.5}
    assert "hybrid_variance_ratio_shape" in fbm_contract_errors(bad3)


def test_write_fbm_receipt_roundtrip(tmp_path: Path) -> None:
    payload = fbm_bench(n=128, n_paths=200, seed=11)
    p = write_fbm_receipt(payload, receipts_dir=tmp_path)
    data = json.loads(p.read_text())
    assert verify_receipt_payload(data)["valid"]
    assert "receipt_sha256" in data
    assert fbm_contract_errors(data) == []


def test_committed_receipt_verifies() -> None:
    repo = Path(__file__).resolve().parents[3]
    receipt = repo / "receipts" / "fbm_circulant.json"
    if not receipt.exists():
        pytest.skip("receipt not committed")
    data = json.loads(receipt.read_text())
    assert verify_receipt_payload(data)["valid"]
    assert lane_contract_errors(data) == []
    assert data["claim"]["ok"]
    assert math.isfinite(data["claim"]["hybrid_variance_ratio"]["median"])
