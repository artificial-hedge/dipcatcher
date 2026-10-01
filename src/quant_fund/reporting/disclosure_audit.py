"""disclosure_audit — every render path must surface the synthetic marker.

Three renderers emit the ``"> SYNTHETIC DATA"`` banner under a synthetic
flag; the evidence report instead records ``synthetic_evidence_not_promotable``
in warnings. Either mechanism is honest — what must never happen is a
synthetic input rendering *unlabeled*, or an evidence report claiming
``status=complete`` on synthetic evidence (the warning forces
``insufficient_evidence``).

This lane renders each path under both flags and pins which marker each
path carries — drift in the disclosure mechanism is an honesty-contract
violation. Sealed ``disclosure_audit.v1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from quant_fund.reporting.regime_performance import (
    build_regime_performance_from_equity,
    regime_performance_markdown,
)
from quant_fund.reporting.report import build_evidence_report, write_report_text
from quant_fund.reporting.tearsheet import build_tearsheet, tearsheet_markdown
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["disclosure_audit", "disclosure_audit_bench"]

_BANNER = "SYNTHETIC DATA"
_MARKER = "synthetic_evidence_not_promotable"


def _equity(n: int = 60) -> pl.DataFrame:
    start = pl.datetime(2020, 1, 1, time_unit="us", time_zone="UTC")
    return pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start, end=start + pl.duration(days=n - 1), interval="1d", eager=True
            ),
            "nav": np.cumprod(1 + np.random.default_rng(3).normal(0, 0.01, n)),
        }
    )


def disclosure_audit() -> dict[str, Any]:
    results: dict[str, Any] = {}
    equity = _equity()

    sheet_s = build_tearsheet(equity, synthetic=True)
    sheet_r = build_tearsheet(equity, synthetic=False)
    results["tearsheet"] = {
        "banner_on_synthetic": _BANNER in tearsheet_markdown(sheet_s),
        "no_banner_on_real": _BANNER not in tearsheet_markdown(sheet_r),
    }

    rep_s = build_regime_performance_from_equity(equity, synthetic=True)
    rep_r = build_regime_performance_from_equity(equity, synthetic=False)
    results["regime_performance"] = {
        "banner_on_synthetic": _BANNER in regime_performance_markdown(rep_s),
        "no_banner_on_real": _BANNER not in regime_performance_markdown(rep_r),
    }

    ev_s = build_evidence_report(
        candidates={"m": 1.0},
        provenance={
            "data_source": "SYNTHETIC",
            "artifact_sha256": "a" * 64,
            "manifest_valid": True,
        },
        health={"status": "ok"},
        promotion={"promote": True},
    )
    results["evidence_report_synthetic"] = {
        "marker_in_warnings": _MARKER in ev_s["warnings"],
        "status_not_complete": ev_s["status"] != "complete",
        "rendered_text_discloses": _MARKER in write_report_text(ev_s),
    }

    ev_none = build_evidence_report(candidates={"m": 1.0})
    results["evidence_report_missing_provenance"] = {
        "warns": "provenance_missing" in ev_none["warnings"],
        "status": ev_none["status"],
    }

    ev_r = build_evidence_report(
        candidates={"m": 1.0},
        provenance={"data_source": "exchange", "artifact_sha256": "a" * 64, "manifest_valid": True},
        health={"status": "ok"},
        promotion={"promote": True},
    )
    results["evidence_report_clean"] = {
        "no_synthetic_marker": _MARKER not in ev_r["warnings"],
        "status": ev_r["status"],
    }
    return results


def disclosure_audit_bench() -> dict[str, Any]:
    r = disclosure_audit()
    ok = (
        r["tearsheet"]["banner_on_synthetic"]
        and r["tearsheet"]["no_banner_on_real"]
        and r["regime_performance"]["banner_on_synthetic"]
        and r["regime_performance"]["no_banner_on_real"]
        and r["evidence_report_synthetic"]["marker_in_warnings"]
        and r["evidence_report_synthetic"]["status_not_complete"]
        and r["evidence_report_synthetic"]["rendered_text_discloses"]
        and r["evidence_report_missing_provenance"]["warns"]
        and r["evidence_report_missing_provenance"]["status"] == "insufficient_evidence"
        and r["evidence_report_clean"]["no_synthetic_marker"]
    )
    payload: dict[str, Any] = {
        "kind": "disclosure_audit",
        "schema": "disclosure_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Every reporting surface discloses synthetic input: banners on "
            "tearsheet/regime-performance renderers, warning-marker on the "
            "evidence report, and a synthetic report can never reach "
            "status=complete."
            if ok
            else f"DISCLOSURE GAP: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
