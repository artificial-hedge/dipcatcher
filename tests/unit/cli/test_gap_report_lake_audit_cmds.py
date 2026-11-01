"""Unit tests for ``quant_fund.cli.report_cmds``, ``lake_cmds`` and ``audit_cmds``.

Three small command modules share one property worth pinning: **they fail
closed with a non-zero exit code and a reason**, instead of exiting zero on a
broken input or silently accepting a malformed record.

* ``verify-research`` on an unreadable artifact reports ``valid: false`` with
  the reason and exits 1 — it never exits 0 on a file it could not read.
* ``lineage show`` for an unknown dataset exits 1 with the dataset named.
* ``lake quality`` writes a machine-readable report even when a threshold
  fails, and exits 1 when ``--enforce`` is on.
* ``audit-record`` refuses to run without a signing key, and refuses to run
  with a key but nothing to record.

The audit ledger itself is never written: every failing path is exercised
before a ledger entry could be produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app

RUNNER = CliRunner()


# --------------------------------------------------------------------------- #
# verify-research
# --------------------------------------------------------------------------- #


def test_verify_research_fails_closed_on_an_unreadable_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    result = RUNNER.invoke(app, ["verify-research", str(missing)])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["valid"] is False
    assert payload["errors"]


def test_verify_research_fails_closed_on_malformed_json(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    result = RUNNER.invoke(app, ["verify-research", str(broken)])
    assert result.exit_code == 1
    assert json.loads(result.output)["valid"] is False


def test_verify_research_reports_the_path_it_checked(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    result = RUNNER.invoke(app, ["verify-research", str(missing)])
    assert json.loads(result.output)["path"] == str(missing)


# --------------------------------------------------------------------------- #
# lineage / lake
# --------------------------------------------------------------------------- #


def test_lineage_show_fails_closed_for_an_unknown_dataset(tmp_path: Path) -> None:
    result = RUNNER.invoke(app, ["lineage", "show", "no-such-dataset", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "no-such-dataset" in result.output


def test_the_lake_and_lineage_sub_apps_are_mounted() -> None:
    groups = {group.name for group in app.registered_groups}
    assert "lake" in groups
    lake = next(g for g in app.registered_groups if g.name == "lake")
    commands = {command.name for command in lake.typer_instance.registered_commands}
    assert {"import", "quality"} <= commands


def test_lake_quality_fails_closed_on_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.parquet"
    result = RUNNER.invoke(app, ["lake", "quality", "--path", str(missing)])
    assert result.exit_code != 0


def test_lake_quality_writes_a_machine_readable_report(tmp_path: Path) -> None:
    """A passing file still produces a JSON report, not just an exit code."""
    import polars as pl

    path = tmp_path / "bars.parquet"
    pl.DataFrame(
        {
            "event_time": [1, 2, 3],
            "log_return": [0.0, 0.001, -0.002],
            "close": [100.0, 100.1, 99.9],
        }
    ).write_parquet(path)
    result = RUNNER.invoke(app, ["lake", "quality", "--path", str(path)])
    assert result.exit_code == 0
    assert json.loads(result.output)["passed"] is True


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #


def test_audit_record_refuses_to_run_without_a_signing_key(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    result = RUNNER.invoke(app, ["audit-record", "--ledger", str(ledger)])
    assert result.exit_code != 0
    assert "pass --signing-key or --generate-key" in result.output


def test_audit_record_refuses_to_run_with_nothing_to_record(tmp_path: Path) -> None:
    """A key without a receipt or paper dir is a usage error, not an empty entry."""
    ledger = tmp_path / "ledger"
    key = tmp_path / "signing.key"
    key.write_text("00" * 32, encoding="utf-8")
    result = RUNNER.invoke(
        app,
        ["audit-record", "--ledger", str(ledger), "--signing-key", str(key)],
    )
    assert result.exit_code != 0
    assert "pass --receipt and/or --paper-dir" in result.output


def test_audit_trace_fails_closed_on_a_missing_ledger(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "audit-trace",
            "--ledger",
            str(tmp_path / "absent-ledger"),
            "--receipt",
            str(receipt),
            "--metric",
            "a.b",
        ],
    )
    assert result.exit_code != 0


def test_the_audit_commands_are_registered_under_their_documented_names() -> None:
    names = {command.name for command in app.registered_commands}
    assert {"verify-ledger", "audit-record", "audit-trace"} <= names


@pytest.mark.parametrize("command", ["verify-ledger", "audit-record", "audit-trace"])
def test_each_audit_command_documents_itself(command: str) -> None:
    """Every audit command must carry a docstring the operator can read."""
    registered = next(c for c in app.registered_commands if c.name == command)
    assert registered.callback is not None, command
    assert registered.callback.__doc__, command
