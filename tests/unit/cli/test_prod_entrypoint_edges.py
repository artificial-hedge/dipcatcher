from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from typer.testing import CliRunner

from quant_fund.cli.forecast_cmds import _parse_asof
from quant_fund.cli.main import app
from quant_fund.utils.cli_errors import DipcatcherTyperGroup

_RUNNER = CliRunner()


def test_naive_cli_date_is_explicitly_utc() -> None:
    assert _parse_asof("2024-01-02T03:04:05") == datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_aware_cli_date_keeps_its_timezone() -> None:
    parsed = _parse_asof("2024-01-02T03:04:05+05:30")

    assert parsed is not None
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset().total_seconds() == 5.5 * 60 * 60


def test_backtest_rejects_unknown_engine_before_config_io(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    result = _RUNNER.invoke(app, ["backtest", "--engine", "unknown", "--config", str(missing)])

    assert result.exit_code == 2
    assert "--engine must be 'ref' or 'fast'" in result.output
    assert "Traceback" not in result.output
    assert not missing.exists()


def test_broken_pipe_exits_without_a_traceback() -> None:
    command_app = typer.Typer(cls=DipcatcherTyperGroup)

    @command_app.command("pipe")
    def pipe() -> None:
        raise BrokenPipeError

    @command_app.command("other")
    def other() -> None:
        pass

    result = _RUNNER.invoke(command_app, ["pipe"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
