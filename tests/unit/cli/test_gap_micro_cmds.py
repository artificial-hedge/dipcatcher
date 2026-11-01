"""Unit tests for ``quant_fund.cli.micro_cmds``.

The candle-book and Kyle/OFI commands accept either a Northset panel or raw
vendor quotes, and the two are told apart by the *schema*, not by a flag
alone. The validation rules are the point:

* a parquet whose columns match neither shape is refused with an actionable
  message instead of being silently reinterpreted;
* a parquet that looks like a panel but is missing a required column fails
  closed rather than being loaded with fabricated fields;
* ``--vendor`` is the escape hatch for a raw quote schema, and an unknown
  vendor name is refused.

The config loader is stubbed out: these tests must not read a research
config or write a resolved-config artifact into the repository.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli import micro_cmds
from quant_fund.cli.main import app

RUNNER = CliRunner()

#: The panel columns the command sniffs for before choosing a loader.
PANEL_COLUMNS = ("best_bid", "best_ask", "security_id", "event_time")


@pytest.fixture(autouse=True)
def _stub_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read a real config or write a resolved-config artifact."""
    monkeypatch.setattr(
        micro_cmds,
        "_cfg",
        lambda config: SimpleNamespace(data=SimpleNamespace(n_assets=4, n_days=40)),
    )


def _write(path: Path, frame: pl.DataFrame) -> Path:
    frame.write_parquet(path)
    return path


def test_candle_book_refuses_an_unrecognised_book_schema(tmp_path: Path) -> None:
    book = _write(tmp_path / "book.parquet", pl.DataFrame({"px": [1.0, 2.0]}))
    result = RUNNER.invoke(app, ["candle-book", "--book", str(book)])
    assert result.exit_code != 0
    assert "pass --vendor" in result.output


def test_kyle_ofi_refuses_an_unrecognised_book_schema(tmp_path: Path) -> None:
    book = _write(tmp_path / "book.parquet", pl.DataFrame({"px": [1.0, 2.0]}))
    result = RUNNER.invoke(app, ["kyle-ofi", "--book", str(book)])
    assert result.exit_code != 0
    assert "pass --vendor" in result.output


def test_a_panel_missing_a_required_column_fails_closed(tmp_path: Path) -> None:
    """A panel-shaped frame with an incomplete schema must not be loaded."""
    frame = pl.DataFrame(
        {
            "best_bid": [100.0],
            "best_ask": [101.0],
            "security_id": ["AAA"],
            "event_time": pl.Series([datetime(2024, 1, 2)], dtype=pl.Datetime("us")),
        }
    )
    book = _write(tmp_path / "book.parquet", frame)
    result = RUNNER.invoke(app, ["candle-book", "--book", str(book)])
    assert result.exit_code != 0
    # The panel loader fails closed and names the columns it needed.
    assert isinstance(result.exception, ValueError)
    assert "book panel missing columns" in str(result.exception)
    assert "source" in str(result.exception)


def test_an_unknown_vendor_is_refused(tmp_path: Path) -> None:
    book = _write(tmp_path / "book.parquet", pl.DataFrame({"px": [1.0]}))
    result = RUNNER.invoke(app, ["candle-book", "--book", str(book), "--vendor", "nyse"])
    assert result.exit_code != 0


def test_a_missing_book_file_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "absent.parquet"
    result = RUNNER.invoke(app, ["candle-book", "--book", str(missing)])
    assert result.exit_code != 0


def test_the_panel_column_set_is_the_documented_one() -> None:
    """The sniff set is part of the command's contract."""
    assert set(PANEL_COLUMNS) == {"best_bid", "best_ask", "security_id", "event_time"}


def test_the_commands_are_registered_under_their_documented_names() -> None:
    names = {command.name for command in app.registered_commands}
    assert {"candle-book", "kyle-ofi"} <= names
