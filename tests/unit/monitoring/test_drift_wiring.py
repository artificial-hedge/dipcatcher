"""Family-specific calibration-drift wiring + evidence-report hash-sidecar check.

SYNTHETIC windows only (correctness tests, never market evidence). Proves
per-family calibration drift surfaces through ``model_health_report`` and that
the ``/monitoring/drift`` evidence-report hash-sidecar check remains fail-closed.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from quant_fund.monitoring.drift import model_health_report, verify_evidence_report_sidecar


def _windows(shift: float = 0.0, n: int = 40):
    rng = np.random.default_rng(0)
    ref_p = rng.uniform(0.3, 0.7, n)
    ref_y = (rng.uniform(0, 1, n) < ref_p).astype(float)
    cur_p = np.clip(ref_p + shift, 0.0, 1.0)
    cur_y = ref_y
    return ref_p, ref_y, cur_p, cur_y


def test_family_calibration_wired_into_health_report() -> None:
    feats = {"f1": np.arange(20.0)}
    report = model_health_report(
        feats,
        dict(feats),
        family_calibration_windows={"garch": _windows(0.0), "hmm": _windows(0.0)},
    )
    assert "family_calibration" in report
    assert set(report["family_calibration"]) == {"garch", "hmm"}
    assert report["family_calibration"]["garch"]["alert"] is False
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True


def test_family_calibration_alerts_per_family() -> None:
    feats = {"f1": np.arange(20.0)}
    report = model_health_report(
        feats,
        dict(feats),
        family_calibration_windows={"good": _windows(0.0), "bad": _windows(0.6)},
        calibration_alert_delta=0.05,
    )
    assert report["family_calibration"]["bad"]["alert"] is True
    assert report["family_calibration"]["good"]["alert"] is False
    assert "calibration:bad" in report["alerts"]
    assert "calibration:good" not in report["alerts"]
    assert report["status"] == "alert"


def test_family_calibration_length_mismatch_fails_closed() -> None:
    feats = {"f1": np.arange(20.0)}
    bad = (np.arange(5.0), np.arange(4.0), np.arange(5.0), np.arange(5.0))
    report = model_health_report(feats, dict(feats), family_calibration_windows={"x": bad})
    assert report["status"] == "invalid"
    assert any(e.startswith("x:calibration_window_length_mismatch") for e in report["errors"])


def test_global_calibration_still_supported() -> None:
    feats = {"f1": np.arange(20.0)}
    report = model_health_report(feats, dict(feats), calibration_windows=_windows(0.0))
    assert report["calibration"] is not None
    assert report["calibration"]["alert"] is False


# -- evidence-report hash-sidecar check (preserved fail-closed) --


def _write_report(tmp_path, tamper: bool = False):
    report = {"status": "complete", "warnings": []}
    p = tmp_path / "evidence_report.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    sidecar = tmp_path / "evidence_report.json.sha256"
    sidecar.write_text(hashlib.sha256(p.read_bytes()).hexdigest(), encoding="ascii")
    if tamper:
        p.write_text(json.dumps({"status": "forged", "warnings": []}), encoding="utf-8")
    return p


def test_evidence_report_sidecar_valid(tmp_path) -> None:
    verdict = verify_evidence_report_sidecar(_write_report(tmp_path))
    assert verdict["valid"] is True
    assert verdict["status"] == "complete"


def test_evidence_report_sidecar_tamper_detected(tmp_path) -> None:
    verdict = verify_evidence_report_sidecar(_write_report(tmp_path, tamper=True))
    assert verdict["valid"] is False
    assert verdict["reason"] == "hash_mismatch"


def test_evidence_report_sidecar_missing_is_invalid(tmp_path) -> None:
    p = tmp_path / "evidence_report.json"
    p.write_text("{}", encoding="utf-8")  # no sidecar written
    verdict = verify_evidence_report_sidecar(p)
    assert verdict["valid"] is False
