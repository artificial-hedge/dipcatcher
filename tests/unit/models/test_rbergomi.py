"""Contract + correctness pins for the rough-Bergomi lane."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant_fund.models.rbergomi import (
    RBERGOMI_SCHEMA,
    RBergomiParams,
    hybrid_volterra,
    rbergomi_paths,
    rbergomi_smile,
    rough_vol_bench,
    rough_vol_contract_errors,
    write_rough_vol_receipt,
)
from quant_fund.research.lane_contracts import lane_contract_errors
from quant_fund.research.receipt_v2 import verify_receipt_payload

_PARAMS = RBergomiParams(1.0, 0.04, 0.10, 1.9, -0.7, 0.25)


def test_hybrid_volterra_is_seeded_deterministic() -> None:
    a, _ = hybrid_volterra(60, 0.10, 0.25 / 60, 200, 11)
    b, _ = hybrid_volterra(60, 0.10, 0.25 / 60, 200, 11)
    c, _ = hybrid_volterra(60, 0.10, 0.25 / 60, 200, 12)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_volterra_variance_near_theory() -> None:
    # at moderate H the kappa=1 hybrid scheme is near-exact
    h = 0.30
    wh, _ = hybrid_volterra(200, h, 1.0 / 200, 4000, 7)
    tt = np.arange(0, 200) / 200.0
    ratio = wh.var(axis=1) * (2.0 * h) / np.maximum(tt, 1e-12) ** (2.0 * h)
    mid = ratio[len(ratio) // 4 :]
    assert mid.min() > 0.9 and mid.max() < 1.1


def test_paths_martingale_and_positive() -> None:
    s_t = rbergomi_paths(_PARAMS, 80, 4000, 42)
    assert np.all(s_t > 0.0)
    assert abs(s_t.mean() / _PARAMS.s0 - 1.0) < 0.02


def test_smile_has_negative_skew() -> None:
    logks = np.linspace(-1.5, 1.5, 9) * np.sqrt(_PARAMS.xi0 * _PARAMS.t)
    iv = rbergomi_smile(_PARAMS, _PARAMS.s0 * np.exp(logks), 80, 4000, 42)
    slope = np.polyfit(logks, iv, 1)[0]
    assert slope < 0.0  # rho < 0 -> smirk
    assert np.all(np.isfinite(iv))


def test_rough_vol_bench_small() -> None:
    payload = rough_vol_bench(n_paths=2000, n_steps=60, seed=91)
    assert payload["schema"] == RBERGOMI_SCHEMA
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["research_only"] is True
    assert payload["live_pnl_claim"] is False
    claim = payload["claim"]
    assert claim["n_probes"] == len(claim["results"])
    assert claim["results"]["martingale"]
    assert claim["results"]["volterra_variance_shape"]


def test_committed_receipt_dispatches_contract() -> None:
    path = Path("receipts/rough_vol.json")
    if not path.exists():
        pytest.skip("receipt not generated yet")
    payload = json.loads(path.read_text())
    assert payload["schema"] == RBERGOMI_SCHEMA
    assert verify_receipt_payload(payload)["valid"]
    assert lane_contract_errors(payload) == []


def test_contract_catches_incoherent_claim() -> None:
    payload = rough_vol_bench(n_paths=1000, n_steps=50, seed=5)
    bad = json.loads(json.dumps(payload))
    bad["claim"]["n_passed"] = 0
    assert "n_passed_mismatch" in rough_vol_contract_errors(bad)
    bad2 = json.loads(json.dumps(payload))
    bad2["claim"]["skew_exponent"]["measured"] = "big"
    assert "skew_exponent_shape" in rough_vol_contract_errors(bad2)
    bad3 = json.loads(json.dumps(payload))
    bad3["research_only"] = False
    assert "research_only" in rough_vol_contract_errors(bad3)


def test_write_receipt_seals_and_verifies(tmp_path: Path) -> None:
    payload = rough_vol_bench(n_paths=1000, n_steps=50, seed=5)
    out = write_rough_vol_receipt(payload, receipts_dir=tmp_path)
    sealed = json.loads(out.read_text())
    assert sealed["receipt_sha256"]
    assert verify_receipt_payload(sealed)["valid"]
    with pytest.raises(ValueError):
        write_rough_vol_receipt({**payload, "kind": "other"}, receipts_dir=tmp_path)
