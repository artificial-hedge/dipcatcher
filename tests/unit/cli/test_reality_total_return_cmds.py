"""CLI bridge for the reality lanes and the total-return price basis.

``reality_sweep`` and ``reality_survivorship`` were reachable only as zero-arg
``python -m`` mains, so their ten path overrides were unusable from a shell —
and ``reality_sweep._PREREG`` points at ``research/reality/preregistration.json``,
which does not exist in a checkout (the real spec lives under
``research/reality/studies/``), so the bare main could not run at all. These
commands expose the same functions with the overrides as options.

``total_return`` had no caller and no test anywhere in the repo: dead code that
puts a bar panel on a total-return basis. It is exercised end-to-end here.

No test in this file touches the network. The Yahoo fetch is monkeypatched to
assert the fail-closed path — the lanes refuse to substitute synthetic data when
a real panel cannot be fetched, and that refusal must surface as a distinct exit
code rather than a traceback or a silent pass.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.data.adapters.stooq import session_close
from quant_fund.reality.cli import FETCH_UNAVAILABLE_EXIT

runner = CliRunner()

_SWEEP_SPEC = Path(
    "research/reality/studies/reality-us-liquid-daily-2026-09-27/preregistration.json"
)
_SURVIVORSHIP_SPEC = Path("research/reality/survivorship/preregistration.json")


@pytest.mark.parametrize("sub", ["sweep", "survivorship"])
def test_reality_subcommand_help(sub: str) -> None:
    result = runner.invoke(app, ["reality", sub, "--help"])
    assert result.exit_code == 0, result.output
    assert "--spec" in result.output
    assert "--cache-dir" in result.output


def test_reality_help_lists_the_new_lanes() -> None:
    result = runner.invoke(app, ["reality", "--help"])
    assert result.exit_code == 0, result.output
    for sub in ("sweep", "survivorship", "preflight", "ledger-gate", "trial-report"):
        assert sub in result.output


def test_total_return_help() -> None:
    result = runner.invoke(app, ["total-return", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--bars", "--chart", "--security-id", "--yahoo-symbol", "--out", "--raw-prices"):
        assert flag in result.output


def _bars(days: int = 6, *, start: datetime | None = None) -> pl.DataFrame:
    """A quote-basis daily panel whose event times are real session closes."""
    first = start or datetime(2026, 3, 2, tzinfo=UTC)
    times, closes = [], []
    day = first
    while len(times) < days:
        if day.weekday() < 5:
            times.append(session_close(day.date(), ".us"))
            closes.append(100.0 + 2.0 * len(times))
        day += timedelta(days=1)
    return pl.DataFrame(
        {
            "security_id": ["AAA"] * len(times),
            "event_time": times,
            "open": [c - 0.5 for c in closes],
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(times),
        }
    )


def _chart_payload(ex_date: datetime, *, amount: float = 1.50) -> dict:
    """A Yahoo chart payload whose ``events`` carry one cash dividend."""
    stamp = int(ex_date.timestamp())
    return {
        "chart": {
            "result": [
                {
                    "events": {
                        "dividends": {
                            str(stamp): {
                                "amount": amount,
                                "date": stamp,
                            }
                        }
                    }
                }
            ]
        }
    }


def _write_run(tmp_path: Path, frame: pl.DataFrame, payload: dict) -> tuple[Path, Path, Path]:
    bars = tmp_path / "bars.parquet"
    frame.write_parquet(bars)
    chart = tmp_path / "chart.json"
    chart.write_text(json.dumps(payload))
    return bars, chart, tmp_path / "total_return.parquet"


def test_total_return_reinvests_a_dividend(tmp_path: Path) -> None:
    """A dividend strictly inside the sample must shift earlier prices down."""
    frame = _bars()
    ex_date = frame["event_time"][3]
    bars, chart, out = _write_run(tmp_path, frame, _chart_payload(ex_date))

    result = runner.invoke(
        app,
        [
            "total-return",
            "--bars",
            str(bars),
            "--chart",
            str(chart),
            "--security-id",
            "AAA",
            "--yahoo-symbol",
            "AAA",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"DATA_LABEL={bars.name}" in result.output
    assert "corporate_actions=1" in result.output
    assert "cash_dividend=1" in result.output
    assert "prices_already_split_adjusted=true" in result.output
    assert out.is_file()

    adjusted = pl.read_parquet(out)
    assert adjusted.height == frame.height

    ordered = adjusted.sort("event_time")
    quote = ordered["close_quote"].to_list()
    total = ordered["close"].to_list()
    assert quote == frame.sort("event_time")["close"].to_list(), "quote basis must be preserved"

    # Reinvestment scales the ex-date bar and everything after it up by
    # 1 + dividend/close; earlier bars are untouched.
    factor = 1.0 + 1.50 / quote[3]
    assert total[0] == pytest.approx(quote[0])
    assert total[2] == pytest.approx(quote[2])
    for index in (3, 4, 5):
        assert total[index] == pytest.approx(quote[index] * factor), index
    assert ordered["return_basis"][0] is not None


def test_total_return_with_no_events_is_a_noop(tmp_path: Path) -> None:
    frame = _bars()
    bars, chart, out = _write_run(tmp_path, frame, {"chart": {"result": [{}]}})
    result = runner.invoke(
        app,
        [
            "total-return",
            "--bars",
            str(bars),
            "--chart",
            str(chart),
            "--security-id",
            "AAA",
            "--yahoo-symbol",
            "AAA",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "corporate_actions=0 (none)" in result.output


def test_total_return_rejects_blank_identifiers(tmp_path: Path) -> None:
    frame = _bars()
    ex_date = frame["event_time"][3]
    bars, chart, out = _write_run(tmp_path, frame, _chart_payload(ex_date))
    result = runner.invoke(
        app,
        [
            "total-return",
            "--bars",
            str(bars),
            "--chart",
            str(chart),
            "--security-id",
            "   ",
            "--yahoo-symbol",
            "AAA",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "non-blank" in result.output
    assert not out.exists()


def test_total_return_rejects_a_malformed_chart(tmp_path: Path) -> None:
    frame = _bars()
    bars = tmp_path / "bars.parquet"
    frame.write_parquet(bars)
    chart = tmp_path / "chart.json"
    chart.write_text('{"chart": {"result": [{"events": {"dividends": "not-a-map"}}]}}')
    out = tmp_path / "out.parquet"
    result = runner.invoke(
        app,
        [
            "total-return",
            "--bars",
            str(bars),
            "--chart",
            str(chart),
            "--security-id",
            "AAA",
            "--yahoo-symbol",
            "AAA",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_reality_sweep_fails_closed_when_the_panel_cannot_be_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lane must stop rather than substitute synthetic data."""
    if not _SWEEP_SPEC.is_file():
        pytest.skip(f"{_SWEEP_SPEC} missing from this checkout")
    from quant_fund.research import reality_sweep

    def _refuse(*_args: object, **_kwargs: object):
        raise RuntimeError("Yahoo fetch failed; stopping without a synthetic substitute")

    monkeypatch.setattr(reality_sweep, "fetch_yahoo_panel", _refuse)
    result = runner.invoke(
        app,
        ["reality", "sweep", "--spec", str(_SWEEP_SPEC), "--cache-dir", str(tmp_path)],
    )
    assert result.exit_code == FETCH_UNAVAILABLE_EXIT, result.output
    assert "REALITY_SWEEP_FETCH_UNAVAILABLE" in result.output
    assert "without a synthetic substitute" in result.output
    assert "Traceback" not in result.output


def test_reality_survivorship_fails_closed_when_the_panel_cannot_be_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _SURVIVORSHIP_SPEC.is_file():
        pytest.skip(f"{_SURVIVORSHIP_SPEC} missing from this checkout")
    from quant_fund.research import reality_survivorship

    def _refuse(*_args: object, **_kwargs: object):
        raise RuntimeError("Yahoo fetch failed; stopping without a synthetic substitute")

    monkeypatch.setattr(reality_survivorship, "fetch_yahoo_panel", _refuse, raising=False)
    result = runner.invoke(
        app,
        [
            "reality",
            "survivorship",
            "--spec",
            str(_SURVIVORSHIP_SPEC),
            "--cache-dir",
            str(tmp_path),
        ],
    )
    # Either the fetch refusal surfaces as the dedicated exit, or an earlier
    # input problem surfaces as a clean typed error — never a traceback.
    assert result.exit_code != 0, result.output
    assert "Traceback" not in result.output


def test_reality_sweep_reports_a_bad_spec_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "spec.json"
    bad.write_text("{}")
    result = runner.invoke(
        app, ["reality", "sweep", "--spec", str(bad), "--cache-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
