"""Fail-closed error handling in ``quant_fund.reporting``.

A research report that says ``complete`` is a claim: it asserts the evidence
was present, well-formed, and approved. These tests pin the cases where that
claim must *not* be made:

- A section that is **absent** is recorded as a missing-section warning.
- A section that is **present but malformed** (not a mapping) is a defect in
  the evidence, and must be flagged — a garbage promotion receipt must never
  read as an approved one.
- The written report's ``.sha256`` sidecar must match its bytes, and the
  drift-side verifier must agree.
- A tearsheet over unusable input must fail closed to an honest stub, never
  to a report with invented numbers.

Every fixture is a ``tmp_path`` file. No receipt store, evidence index, or
real data directory is read or written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from quant_fund.monitoring.drift import verify_evidence_report_sidecar
from quant_fund.reporting.report import (
    build_evidence_report,
    latest_report_dir,
    write_evidence_report,
    write_report,
    write_report_text,
)
from quant_fund.reporting.tearsheet import (
    build_tearsheet,
    period_returns_table,
    tearsheet_markdown,
    write_tearsheet_md,
)


def _clean_provenance() -> dict[str, object]:
    return {
        "data_source": "vendor",
        "artifact_sha256": "a" * 64,
        "manifest_valid": True,
    }


# --------------------------------------------------------------------------- #
# build_evidence_report — missing sections
# --------------------------------------------------------------------------- #


def test_report_is_complete_only_when_everything_is_clean() -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert report["status"] == "complete"
    assert report["warnings"] == []
    assert report["schema"] == "evidence_report.v1"


def test_report_records_a_missing_candidate_section() -> None:
    report = build_evidence_report(candidates={})
    assert report["status"] == "insufficient_evidence"
    assert "candidate_metrics_missing" in report["warnings"]


def test_report_records_missing_provenance_health_and_promotion() -> None:
    report = build_evidence_report(candidates={"ranker": {"mean_ic": 0.1}})
    warnings = report["warnings"]
    assert "provenance_missing" in warnings
    assert "health_report_missing" in warnings
    assert "promotion_receipt_missing" in warnings


def test_report_flags_an_unverified_artifact_manifest() -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance={"data_source": "vendor", "artifact_sha256": "a" * 64},
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert "artifact_manifest_not_verified" in report["warnings"]
    assert report["status"] == "insufficient_evidence"


def test_report_flags_a_short_artifact_hash() -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance={"data_source": "vendor", "artifact_sha256": "abc", "manifest_valid": True},
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert "artifact_hash_missing_or_invalid" in report["warnings"]


def test_report_flags_an_unapproved_promotion() -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion={"promote": False},
    )
    assert "promotion_not_approved" in report["warnings"]
    assert report["status"] == "insufficient_evidence"


def test_report_flags_a_health_report_that_is_not_ok() -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "degraded"},
        promotion={"promote": True},
    )
    assert "health_not_ok" in report["warnings"]


def test_report_labels_synthetic_evidence_as_not_promotable() -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance={
            "data_source": "SYNTHETIC",
            "artifact_sha256": "a" * 64,
            "manifest_valid": True,
        },
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert "synthetic_evidence_not_promotable" in report["warnings"]
    assert report["status"] == "insufficient_evidence"


def test_report_is_never_a_live_pnl_claim() -> None:
    report = build_evidence_report(candidates={})
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False


# --------------------------------------------------------------------------- #
# build_evidence_report — malformed sections (the fail-open this fixes)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("malformed", ["yes", 42, ["promote"], True])
def test_a_malformed_promotion_receipt_is_never_approved(malformed: object) -> None:
    """A non-mapping promotion section must not read as an approved receipt."""
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion=malformed,  # type: ignore[arg-type]
    )
    assert report["status"] == "insufficient_evidence"
    assert "promotion_section_malformed" in report["warnings"]
    assert "promotion_not_approved" in report["warnings"]


@pytest.mark.parametrize("malformed", ["ok", 42, ["status"], True])
def test_a_malformed_health_report_is_never_ok(malformed: object) -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health=malformed,  # type: ignore[arg-type]
    )
    assert report["status"] == "insufficient_evidence"
    assert "health_section_malformed" in report["warnings"]
    assert "health_not_ok" in report["warnings"]


@pytest.mark.parametrize("malformed", ["vendor", 42, ["data_source"], True])
def test_a_malformed_provenance_section_is_never_verified(malformed: object) -> None:
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=malformed,  # type: ignore[arg-type]
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert report["status"] == "insufficient_evidence"
    assert "provenance_section_malformed" in report["warnings"]
    assert "artifact_manifest_not_verified" in report["warnings"]


def test_a_malformed_section_never_yields_a_complete_report() -> None:
    """The whole point: no combination of malformed sections is ``complete``."""
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance="vendor",
        health="ok",
        promotion="yes",
    )
    assert report["status"] == "insufficient_evidence"
    assert report["warnings"]


def test_a_malformed_section_is_normalised_in_the_report_body() -> None:
    """The report keeps its documented shape even when a section is garbage."""
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion="yes",  # type: ignore[arg-type]
    )
    assert report["promotion"] == {}
    assert isinstance(report["health"], dict)
    assert isinstance(report["provenance"], dict)


def test_an_empty_dict_section_is_still_a_missing_section() -> None:
    """``{}`` is absent, not malformed — the original wording is preserved."""
    report = build_evidence_report(
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance={},
        health={},
        promotion={},
    )
    assert "provenance_missing" in report["warnings"]
    assert "health_report_missing" in report["warnings"]
    assert "promotion_receipt_missing" in report["warnings"]
    assert "provenance_section_malformed" not in report["warnings"]


def test_an_absent_section_is_not_reported_as_malformed() -> None:
    report = build_evidence_report(candidates={"ranker": {"mean_ic": 0.1}})
    assert not any(warning.endswith("_section_malformed") for warning in report["warnings"])


def test_an_absent_section_keeps_its_original_wording() -> None:
    """``None`` means *absent*, so it keeps the pre-existing warning set.

    Only a present-but-garbage section is new behaviour; the absent case must
    not quietly gain or lose warnings.
    """
    report = build_evidence_report(candidates={"ranker": {"mean_ic": 0.1}})
    assert report["warnings"] == [
        "artifact_hash_missing_or_invalid",
        "artifact_manifest_not_verified",
        "health_report_missing",
        "promotion_receipt_missing",
        "provenance_missing",
    ]


# --------------------------------------------------------------------------- #
# write_evidence_report — durable output and its sidecar
# --------------------------------------------------------------------------- #


def test_write_evidence_report_persists_json_and_markdown(tmp_path: Path) -> None:
    paths = write_evidence_report(
        tmp_path,
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["schema"] == "evidence_report.v1"
    assert payload["status"] == "complete"
    assert "Institutional Evidence Report" in paths["markdown"].read_text(encoding="utf-8")


def test_write_evidence_report_sidecar_matches_the_bytes(tmp_path: Path) -> None:
    paths = write_evidence_report(
        tmp_path,
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion={"promote": True},
    )
    sidecar = paths["json"].with_name(f"{paths['json'].name}.sha256")
    assert sidecar.is_file()
    expected = hashlib.sha256(paths["json"].read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="ascii").strip() == expected


def test_written_report_verifies_through_the_drift_gate(tmp_path: Path) -> None:
    """The written report and the monitoring-side verifier must agree."""
    paths = write_evidence_report(
        tmp_path,
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion={"promote": True},
    )
    verdict = verify_evidence_report_sidecar(paths["json"])
    assert verdict["valid"] is True
    assert verdict["reason"] == "verified"


def test_written_report_with_a_malformed_section_does_not_verify_as_complete(
    tmp_path: Path,
) -> None:
    paths = write_evidence_report(
        tmp_path,
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion="yes",  # type: ignore[arg-type]
    )
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["status"] == "insufficient_evidence"
    verdict = verify_evidence_report_sidecar(paths["json"])
    assert verdict["valid"] is True
    assert verdict["status"] == "insufficient_evidence"


def test_write_evidence_report_lands_under_the_latest_report_dir(tmp_path: Path) -> None:
    paths = write_evidence_report(
        tmp_path,
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion={"promote": True},
    )
    assert paths["json"].parent == latest_report_dir(tmp_path)


def test_write_evidence_report_leaves_no_temp_files(tmp_path: Path) -> None:
    write_evidence_report(
        tmp_path,
        candidates={"ranker": {"mean_ic": 0.1}},
        provenance=_clean_provenance(),
        health={"status": "ok"},
        promotion={"promote": True},
    )
    leftovers = [p.name for p in latest_report_dir(tmp_path).iterdir() if p.name.startswith(".")]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# write_report_text / write_report
# --------------------------------------------------------------------------- #


def test_write_report_text_stamps_the_research_only_header() -> None:
    text = write_report_text(build_evidence_report(candidates={}))
    assert "research_only: true" in text
    assert "live_pnl_claim: false" in text


def test_write_report_labels_synthetic_output(tmp_path: Path) -> None:
    path = write_report(
        tmp_path / "r.md",
        "Title",
        {"note": "hello"},
        synthetic=True,
    )
    text = path.read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in text
    assert "# Title" in text


def test_write_report_omits_the_synthetic_banner_for_real_data(tmp_path: Path) -> None:
    path = write_report(tmp_path / "r.md", "Title", {"note": "hello"}, synthetic=False)
    assert "SYNTHETIC DATA" not in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# tearsheet — fail closed on unusable input
# --------------------------------------------------------------------------- #


def _equity(n: int = 40) -> pl.DataFrame:
    from datetime import UTC, datetime, timedelta

    base = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "event_time": [base + timedelta(days=i) for i in range(n)],
            "nav": [1_000_000.0 * (1.001**i) for i in range(n)],
        }
    )


def test_tearsheet_over_empty_equity_is_an_empty_stub() -> None:
    sheet = build_tearsheet(pl.DataFrame({"event_time": [], "nav": []}))
    assert sheet["summary"]["status"] == "empty_or_short"
    assert sheet["summary"]["n_bars"] == 0
    assert sheet["live_pnl_claim"] is False
    assert sheet["research_only"] is True


def test_tearsheet_over_a_single_bar_is_an_empty_stub() -> None:
    sheet = build_tearsheet(pl.DataFrame({"nav": [1.0]}))
    assert sheet["summary"]["status"] == "empty_or_short"


def test_tearsheet_without_a_nav_column_raises() -> None:
    with pytest.raises(ValueError, match="nav"):
        build_tearsheet(pl.DataFrame({"event_time": [1, 2]}))


def test_tearsheet_with_non_finite_nav_breaks_the_chain() -> None:
    """A non-finite interior bar must not be bridged into one big return."""
    frame = pl.DataFrame(
        {
            "event_time": [1, 2, 3, 4],
            "nav": [100.0, float("nan"), 110.0, 120.0],
        }
    )
    sheet = build_tearsheet(frame)
    # 100 -> nan -> 110 is not a measurable 1-bar return; 110 -> 120 is.
    assert sheet["summary"]["n_bars"] == 3


def test_tearsheet_summary_reports_n_bars_and_returns() -> None:
    sheet = build_tearsheet(_equity(20))
    assert sheet["summary"]["n_bars"] == 20
    assert sheet["summary"]["total_return"] > 0
    assert "drawdown" in sheet
    assert "risk" in sheet


def test_tearsheet_period_returns_are_bucketed_by_month() -> None:
    table = period_returns_table(_equity(40))
    assert table
    assert all(len(key) == 7 and key[4] == "-" for key in table)


def test_period_returns_table_without_event_time_is_empty() -> None:
    assert period_returns_table(pl.DataFrame({"nav": [1.0, 2.0]})) == {}


def test_period_returns_table_on_a_short_frame_is_empty() -> None:
    assert period_returns_table(pl.DataFrame({"event_time": [1], "nav": [1.0]})) == {}


def test_tearsheet_markdown_renders_the_summary() -> None:
    sheet = build_tearsheet(_equity(20))
    markdown = tearsheet_markdown(sheet)
    assert "# Tearsheet" in markdown
    assert "Research diagnostic only" in markdown
    assert "## Summary" in markdown


def test_tearsheet_markdown_marks_synthetic_output() -> None:
    markdown = tearsheet_markdown(build_tearsheet(_equity(20), synthetic=True))
    assert "SYNTHETIC DATA" in markdown


def test_write_tearsheet_md_persists_the_markdown(tmp_path: Path) -> None:
    path = write_tearsheet_md(tmp_path / "t.md", build_tearsheet(_equity(20)))
    assert path.is_file()
    assert "# Tearsheet" in path.read_text(encoding="utf-8")
