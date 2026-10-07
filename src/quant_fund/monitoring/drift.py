"""Feature / prediction drift. PSI is optional, not the only metric.

Family-specific calibration drift is wired into :func:`model_health_report`
so each model family gets its own Brier-calibration delta/alert rather than one
global number. :func:`verify_evidence_report_sidecar` preserves the
fail-closed evidence-report hash-sidecar check used by ``/monitoring/drift``:
an evidence report whose bytes no longer match its ``.sha256`` sidecar is
``invalid``/``hash_mismatch``, never silently trusted. ``live_pnl_claim=False``
and ``research_only=True`` throughout.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.base import load_joblib_artifact


def drift_report(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
    *,
    psi_threshold: float = 0.25,
    mean_shift_threshold: float | None = None,
    bins: int = 10,
) -> dict[str, float | bool | str]:
    """Return an auditable drift decision with explicit sample sufficiency.

    PSI is only meaningful when both windows support the requested bins. A
    short or empty window therefore produces ``insufficient_data`` rather
    than a false all-clear. Mean shift is optional and remains a diagnostic.
    """
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    n_reference = int(np.isfinite(r).sum())
    n_current = int(np.isfinite(c).sum())
    value = psi(r, c, bins=bins)
    shift = mean_shift(r, c)
    sufficient = bool(np.isfinite(value))
    psi_alert = bool(sufficient and value >= float(psi_threshold))
    mean_alert = bool(
        mean_shift_threshold is not None
        and np.isfinite(shift)
        and abs(shift) >= float(mean_shift_threshold)
    )
    return {
        "status": "alert"
        if psi_alert or mean_alert
        else ("ok" if sufficient else "insufficient_data"),
        "psi": float(value),
        "mean_shift": float(shift),
        "psi_threshold": float(psi_threshold),
        "n_reference": float(n_reference),
        "n_current": float(n_current),
        "sufficient_data": sufficient,
        "alert": psi_alert or mean_alert,
    }


def psi(reference: NDArray[np.float64], current: NDArray[np.float64], bins: int = 10) -> float:
    if int(bins) < 2:
        raise ValueError("bins must be >= 2")
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size < bins or c.size < bins:
        return float("nan")
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    # Keep current-window outliers in the outer bins instead of dropping them.
    edges[0] = min(edges[0], float(np.min(c)))
    edges[-1] = max(edges[-1], float(np.max(c)))
    if np.any(np.diff(edges) <= 0.0):
        edges = np.linspace(
            float(min(np.min(r), np.min(c))), float(max(np.max(r), np.max(c))), bins + 1
        )
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    pr, _ = np.histogram(r, bins=edges)
    pc, _ = np.histogram(c, bins=edges)
    pr = np.clip(pr / pr.sum(), 1e-6, 1.0)
    pc = np.clip(pc / pc.sum(), 1e-6, 1.0)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def mean_shift(reference: NDArray[np.float64], current: NDArray[np.float64]) -> float:
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    r = r[np.isfinite(r)]
    c = c[np.isfinite(c)]
    if r.size == 0 or c.size == 0:
        return float("nan")
    return float(np.mean(c) - np.mean(r))


def model_health_report(
    reference_features: Mapping[str, NDArray[np.float64]],
    current_features: Mapping[str, NDArray[np.float64]],
    *,
    artifact_paths: Mapping[str, str | Path] | None = None,
    calibration_windows: tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]
    | None = None,
    family_calibration_windows: Mapping[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]
    | None = None,
    calibration_alert_delta: float = 0.05,
    psi_threshold: float = 0.25,
    min_features: int = 1,
) -> dict[str, Any]:
    """Return one fail-closed artifact and feature-health report.

    This is an operational diagnostic only: it does not promote or halt a
    model. Missing features, invalid artifacts, and insufficient samples are
    explicit failures rather than being treated as a clean report.
    """
    if min_features < 1:
        raise ValueError("min_features must be positive")
    feature_names = sorted(set(reference_features) | set(current_features))
    feature_reports: dict[str, Any] = {}
    errors: list[str] = []
    for name in feature_names:
        if name not in reference_features or name not in current_features:
            errors.append(f"missing_feature:{name}")
            continue
        report = drift_report(
            reference_features[name], current_features[name], psi_threshold=psi_threshold
        )
        feature_reports[name] = report
        if report["status"] == "insufficient_data":
            errors.append(f"insufficient_feature_data:{name}")
    artifact_reports: dict[str, Any] = {}
    for name, raw_path in (artifact_paths or {}).items():
        path = Path(raw_path)
        try:
            load_joblib_artifact(path)
            artifact_reports[name] = {"status": "ok", "path": str(path)}
        except (OSError, ValueError, TypeError) as exc:
            artifact_reports[name] = {"status": "invalid", "path": str(path)}
            errors.append(f"invalid_artifact:{name}:{type(exc).__name__}")
    calibration, calibration_errors = _calibration_block(
        calibration_windows, calibration_alert_delta
    )
    errors.extend(calibration_errors)
    family_calibration, family_errors, family_alerts = _family_calibration(
        family_calibration_windows, calibration_alert_delta
    )
    errors.extend(family_errors)
    alerts = [name for name, report in feature_reports.items() if report.get("alert")]
    if calibration is not None and calibration.get("alert"):
        alerts.append("calibration")
    alerts.extend(family_alerts)
    if len(feature_reports) < min_features:
        errors.append("insufficient_feature_count")
    status = "invalid" if errors else ("alert" if alerts else "ok")
    return {
        "status": status,
        "features": feature_reports,
        "artifacts": artifact_reports,
        "calibration": calibration,
        "family_calibration": family_calibration,
        "alerts": alerts,
        "errors": errors,
        "research_only": True,
        "live_pnl_claim": False,
    }


def _calibration_block(
    windows: tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
    ]
    | None,
    alert_delta: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Brier-calibration delta for one reference/current prob-label pair.

    Returns ``(block, errors)``. ``block`` is ``None`` when no usable data is
    present; length/sample insufficiency is reported in ``errors`` rather than
    being treated as a clean calibration.
    """
    if windows is None:
        return None, []
    reference_prob, reference_label, current_prob, current_label = windows
    rp = np.asarray(reference_prob, dtype=float)
    ry = np.asarray(reference_label, dtype=float)
    cp = np.asarray(current_prob, dtype=float)
    cy = np.asarray(current_label, dtype=float)
    if rp.size != ry.size or cp.size != cy.size:
        return None, ["calibration_window_length_mismatch"]
    rmask = np.isfinite(rp) & np.isfinite(ry)
    cmask = np.isfinite(cp) & np.isfinite(cy)
    if not rmask.any() or not cmask.any():
        return None, ["insufficient_calibration_data"]
    reference_brier = float(np.mean((rp[rmask] - ry[rmask]) ** 2))
    current_brier = float(np.mean((cp[cmask] - cy[cmask]) ** 2))
    delta = current_brier - reference_brier
    return {
        "reference_brier": reference_brier,
        "current_brier": current_brier,
        "delta": delta,
        "alert": bool(delta >= alert_delta),
    }, []


def _family_calibration(
    family_windows: Mapping[
        str,
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ]
    | None,
    alert_delta: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Per-family Brier-calibration drift. Returns ``(blocks, errors, alerts)``."""
    blocks: dict[str, Any] = {}
    errors: list[str] = []
    alerts: list[str] = []
    for family, windows in (family_windows or {}).items():
        block, block_errors = _calibration_block(windows, alert_delta)
        errors.extend(f"{family}:{e}" for e in block_errors)
        blocks[family] = block
        if block is not None and block.get("alert"):
            alerts.append(f"calibration:{family}")
    return blocks, errors, alerts


def verify_evidence_report_sidecar(path: str | Path) -> dict[str, Any]:
    """Fail-closed evidence-report ``.sha256`` sidecar check.

    Mirrors the ``/monitoring/drift`` integrity gate: the report bytes must
    match their sidecar digest or the report is ``invalid``/``hash_mismatch``.
    A missing/unreadable report or sidecar is ``invalid`` (never trusted).
    """
    report_path = Path(path)
    sidecar = report_path.with_name(f"{report_path.name}.sha256")
    if not report_path.is_file():
        return {"status": "invalid", "reason": "missing", "valid": False}
    try:
        expected = sidecar.read_text(encoding="ascii").strip()
        actual = hashlib.sha256(report_path.read_bytes()).hexdigest()
        if expected != actual:
            return {"status": "invalid", "reason": "hash_mismatch", "valid": False}
        evidence = json.loads(report_path.read_text(encoding="utf-8"))
    except ValueError:
        return {"status": "invalid", "reason": "hash_mismatch", "valid": False}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "invalid", "valid": False}
    status = (
        str(evidence.get("status", "insufficient_evidence"))
        if isinstance(evidence, dict)
        else "insufficient_evidence"
    )
    return {
        "status": status,
        "reason": "verified",
        "valid": True,
        "warnings": evidence.get("warnings", []) if isinstance(evidence, dict) else [],
        "live_pnl_claim": False,
        "research_only": True,
    }
