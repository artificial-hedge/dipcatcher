"""Calibration-drift wiring + evidence sidecar verification tests (task 5).

Proves family-calibration drift is wired into ``model_health_report`` —
per-family Brier-calibration alerting (``calibration:<family>``) and fail-closed
length-mismatch handling — and that the ``/monitoring/drift`` evidence-report
hash-sidecar check verifies a report and fails closed on a missing report or a
tampered (hash-mismatched) one.

Binds to the committed ``drift.py`` surface: ``model_health_report`` takes
``calibration_windows``/``family_calibration_windows`` as 4-tuples
``(reference_prob, reference_label, current_prob, current_label)`` and returns
a ``family_calibration`` block; ``verify_evidence_report_sidecar`` mirrors the
``/monitoring/drift`` integrity gate. SYNTHETIC calibration windows only
(correctness tests, never market evidence). ``live_pnl_claim=False`` throughout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from quant_fund.monitoring.drift import model_health_report, verify_evidence_report_sidecar

SEED = 20261007


def _features() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One clean SYNTHETIC feature pair (identical ref/cur -> zero drift)."""
    rng = np.random.default_rng(SEED)
    ref = {"f0": rng.normal(0.0, 1.0, 200)}
    cur = {"f0": ref["f0"].copy()}
    return ref, cur


def _windows(
    shift: float = 0.0, n: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SYNTHETIC (ref_prob, ref_label, cur_prob, cur_label); +shift worsens it."""
    rng = np.random.default_rng(SEED)
    ref_label = rng.integers(0, 2, size=n).astype(float)
    ref_prob = rng.uniform(0.05, 0.95, size=n)
    cur_label = ref_label.copy()
    cur_prob = np.clip(ref_prob + shift, 0.0, 1.0)
    return ref_prob, ref_label, cur_prob, cur_label


def test_family_calibration_wired_into_health_report() -> None:
    ref, cur = _features()
    report = model_health_report(
        ref, cur, family_calibration_windows={"garch": _windows(), "hmm": _windows()}
    )
    assert set(report["family_calibration"]) == {"garch", "hmm"}
    assert report["live_pnl_claim"] is False
    assert report["research_only"] is True


def test_family_calibration_alerts_per_family() -> None:
    ref, cur = _features()
    report = model_health_report(ref, cur, family_calibration_windows={"bad": _windows(0.6)})
    assert report["family_calibration"]["bad"]["alert"] is True
    assert "calibration:bad" in report["alerts"]
    assert report["status"] == "alert"


def test_family_calibration_length_mismatch_fails_closed() -> None:
    ref, cur = _features()
    ref_prob, ref_label, cur_prob, cur_label = _windows()
    mismatched = (ref_prob, ref_label[:-1], cur_prob, cur_label)  # ref length mismatch
    report = model_health_report(ref, cur, family_calibration_windows={"x": mismatched})
    assert report["status"] == "invalid"
    assert "x:calibration_window_length_mismatch" in report["errors"]


def test_global_calibration_still_supported() -> None:
    ref, cur = _features()
    report = model_health_report(ref, cur, calibration_windows=_windows())
    assert report["calibration"] is not None
    assert "reference_brier" in report["calibration"]
    assert report["calibration"]["alert"] is False
    assert report["family_calibration"] == {}


def _write_report(tmp_path: Path, tamper: bool = False) -> Path:
    p = tmp_path / "evidence_report.json"
    p.write_text(json.dumps({"status": "complete", "warnings": []}), encoding="utf-8")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "evidence_report.json.sha256").write_text(digest, encoding="ascii")
    if tamper:
        p.write_text(json.dumps({"status": "forged"}), encoding="utf-8")
    return p


def test_evidence_report_sidecar_valid(tmp_path) -> None:
    verdict = verify_evidence_report_sidecar(_write_report(tmp_path))
    assert verdict["valid"] is True
    assert verdict["status"] == "complete"
    assert verdict["reason"] == "verified"


def test_evidence_report_sidecar_tamper_detected(tmp_path) -> None:
    verdict = verify_evidence_report_sidecar(_write_report(tmp_path, tamper=True))
    assert verdict["valid"] is False
    assert verdict["status"] == "invalid"
    assert verdict["reason"] == "hash_mismatch"


def test_evidence_report_sidecar_missing_is_invalid(tmp_path) -> None:
    verdict = verify_evidence_report_sidecar(tmp_path / "nope.json")
    assert verdict["valid"] is False
    assert verdict["status"] == "invalid"
