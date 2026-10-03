"""Selftest contract — the golden-path smoke must itself be green, honest
about partial coverage, and reachable from the CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from fx1.cli import app
from fx1.selftest import SelftestReport, run_selftest


def test_local_selftest_all_green(tmp_path) -> None:
    """Boots a stub engine + the production app on loopback and walks the
    whole golden path, including a real process restart on the journal."""
    report = run_selftest(state_dir=str(tmp_path / "state"))
    failed = [c.name for c in report.checks if not c.ok]
    assert not failed, failed
    names = {c.name for c in report.checks}
    assert "restart_recovers_job" in names
    assert "inprocess_parity" in names


def test_local_selftest_without_state_dir_skips_restart() -> None:
    report = run_selftest()
    failed = [c.name for c in report.checks if not c.ok]
    assert not failed, failed
    names = {c.name for c in report.checks}
    assert "restart_recovers_job" not in names


def test_report_shape_is_stable() -> None:
    report = SelftestReport(mode="local")
    report.note("probe", True, "d")
    out = report.as_dict()
    assert out["mode"] == "local"
    assert out["ok"] is True
    assert out["checks"] == [{"name": "probe", "ok": True, "detail": "d"}]


def test_report_not_ok_propagates() -> None:
    report = SelftestReport(mode="local")
    report.note("probe", False)
    assert report.as_dict()["ok"] is False


def test_cli_selftest_local(tmp_path) -> None:
    cr = CliRunner().invoke(app, ["harness", "selftest", "--state-dir", str(tmp_path / "s")])
    assert cr.exit_code == 0, cr.output
    payload = json.loads(cr.output)
    assert payload["ok"] is True
    assert payload["mode"] == "local"


def test_cli_selftest_remote_unreachable_reports_failure() -> None:
    cr = CliRunner().invoke(
        app,
        [
            "harness",
            "selftest",
            "--remote",
            "http://127.0.0.1:1",
            "--timeout",
            "2",
        ],
    )
    assert cr.exit_code == 2
    payload = json.loads(cr.output)
    assert payload["ok"] is False
    assert payload["mode"] == "remote"
