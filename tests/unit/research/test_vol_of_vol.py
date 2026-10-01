"""Contract + correctness pins for the vol-of-vol / roughness lane."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.research.lane_contracts import lane_contract_errors
from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.research.vol_of_vol import (
    GBM_H,
    PLANTED_ETA,
    PLANTED_H,
    VOL_OF_VOL_SCHEMA,
    _arm_estimate,
    _rbergomi_returns,
    _synthetic_silver_panel,
    vol_of_vol_bench,
    vol_of_vol_contract_errors,
    write_vol_of_vol_receipt,
)

_BENCH_SMALL = {"seed": 5, "n_obs": 512, "block": 16, "n_paths": 12}
_LAGS = np.arange(1, 81, dtype=np.float64)


def test_rbergomi_driver_seeded_deterministic() -> None:
    a = _rbergomi_returns(PLANTED_ETA, PLANTED_H, 0.04, -0.7, 4096, 4, 11)
    b = _rbergomi_returns(PLANTED_ETA, PLANTED_H, 0.04, -0.7, 4096, 4, 11)
    c = _rbergomi_returns(PLANTED_ETA, PLANTED_H, 0.04, -0.7, 4096, 4, 12)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.isfinite(a).all()
    assert a.shape == (4096, 4)


def test_planted_h_recovery() -> None:
    # pooled RV chain on the shipped rBergomi driver (hybrid_volterra,
    # κ = 1 bias included) recovers H ~ 0.10 within the honest bound
    nu_w = 0.5 * PLANTED_ETA * math.sqrt(2.0 * PLANTED_H)
    arm = _arm_estimate("rbergomi", nu_w, PLANTED_H, 512, 16, 16, 5, _LAGS, 4)
    assert abs(arm["h_hat"] - PLANTED_H) <= 0.15
    assert arm["eta_hat"] > 0.0  # bound calibrated against the bench run
    assert arm["h_hat_q1"] > 0.0 and arm["h_hat_q2"] > 0.0


def test_gbm_contrast() -> None:
    arm = _arm_estimate("lognormal", 0.30, GBM_H, 512, 16, 16, 42, _LAGS, 4)
    assert abs(arm["h_hat"] - GBM_H) <= 0.15


def test_vol_of_vol_bench_payload_coherent() -> None:
    payload = vol_of_vol_bench(silver_path=None, **_BENCH_SMALL)
    assert payload["schema"] == VOL_OF_VOL_SCHEMA
    assert payload["kind"] == "vol_of_vol"
    assert payload["data_label"] == "SYNTHETIC"
    assert payload["research_only"] is True
    assert payload["live_pnl_claim"] is False
    claim = payload["claim"]
    assert claim["n_probes"] == len(claim["results"])
    assert claim["n_passed"] == sum(1 for v in claim["results"].values() if v)
    assert claim["results"]["seeded_determinism"]
    assert claim["results"]["planted_h_recovery"]
    # no forbidden headline-metric keys anywhere in the blob
    from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

    assert family_blob_forbidden_metrics_absent(payload)


def test_silver_panel_arm_on_generated_fallback() -> None:
    payload = vol_of_vol_bench(silver_path=None, **_BENCH_SMALL)
    silver = payload["claim"]["arms"]["silver_panel"]
    assert silver["source"] == "generated_fallback"
    assert silver["n_symbols"] == 36
    assert silver["symbols_failed"] == []
    assert all(math.isfinite(v) for v in silver["h_hat_by_symbol"].values())
    # an explicit fixture frame also feeds the arm
    small = _synthetic_silver_panel(n_syms=6, n_days=240, seed=3)
    payload2 = vol_of_vol_bench(bars=small, **_BENCH_SMALL)
    silver2 = payload2["claim"]["arms"]["silver_panel"]
    assert silver2["source"] == "fixture"
    assert silver2["n_symbols"] == 6


def test_contract_catches_forged_claims() -> None:
    payload = vol_of_vol_bench(silver_path=None, **_BENCH_SMALL)
    bad = json.loads(json.dumps(payload))
    bad["claim"]["n_passed"] = 0
    assert "n_passed_mismatch" in vol_of_vol_contract_errors(bad)
    bad2 = json.loads(json.dumps(payload))
    bad2["claim"]["ok"] = False
    assert "ok_mismatch" in vol_of_vol_contract_errors(bad2)
    bad3 = json.loads(json.dumps(payload))
    bad3["claim"]["arms"]["rbergomi_planted"]["h_hat"] = "rough"
    assert "rbergomi_planted.h_hat" in vol_of_vol_contract_errors(bad3)
    bad4 = json.loads(json.dumps(payload))
    bad4["claim"]["arms"]["rbergomi_planted"]["planted_h"] = 0.5
    assert "rbergomi_planted.planted_h" in vol_of_vol_contract_errors(bad4)
    bad5 = json.loads(json.dumps(payload))
    bad5["claim"]["arms"]["silver_panel"]["n_symbols"] = 1
    assert "silver_panel.n_symbols" in vol_of_vol_contract_errors(bad5)
    bad6 = json.loads(json.dumps(payload))
    bad6["research_only"] = False
    assert "research_only" in vol_of_vol_contract_errors(bad6)


def test_committed_receipt_dispatches_contract() -> None:
    path = Path("receipts/vol_of_vol.json")
    if not path.exists():
        pytest.skip("receipt not generated yet")
    payload = json.loads(path.read_text())
    assert payload["schema"] == VOL_OF_VOL_SCHEMA
    assert verify_receipt_payload(payload)["valid"]
    assert lane_contract_errors(payload) == []


def test_write_receipt_seals_and_verifies(tmp_path: Path) -> None:
    payload = vol_of_vol_bench(silver_path=None, **_BENCH_SMALL)
    out = write_vol_of_vol_receipt(payload, receipts_dir=tmp_path)
    sealed = json.loads(out.read_text())
    assert sealed["receipt_sha256"]
    assert verify_receipt_payload(sealed)["valid"]
    assert vol_of_vol_contract_errors(sealed) == []
    with pytest.raises(ValueError):
        write_vol_of_vol_receipt({**payload, "kind": "other"}, receipts_dir=tmp_path)
