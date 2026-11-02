"""Yahoo Finance v8 daily chart as a PIT-shaped file tape. Not SIP vintages.

Yahoo EOD OHLC is treated as vendor-adjusted (``revision_id=YAHOO_VENDOR_ADJ``).
``available_time`` equals session close. Same honesty contract as the Stooq
writer: scientific scoring only, no champion alias, no live claim.

Stooq EOD is preferred when it returns CSV; Yahoo is the fallback when Stooq
serves a JavaScript proof-of-work wall.

The default chart URL is quote OHLC only. Pass ``events`` to request dividend
and split rows. Reinvestment for research backtests lives in
``quant_fund.research.total_return`` and does not run inside this adapter.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import quote

import polars as pl

from quant_fund.data.adapters.stooq import (
    US_LIQUID,
    USER_AGENT,
    np_finite,
    session_close,
    write_file_lake,
)
from quant_fund.data.concurrent_io import IoError, call_with_retry, map_ordered, pooled_request

SOURCE = "yahoo"
REVISION = "YAHOO_VENDOR_ADJ"
CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={start}&period2={end}&interval=1d&includeAdjustedClose=true"
)

YAHOO_US: tuple[tuple[str, str], ...] = tuple((sid, sid) for sid, _stooq in US_LIQUID)
YAHOO_UK: tuple[tuple[str, str], ...] = (
    ("VOD", "VOD.L"),
    ("BP", "BP.L"),
    ("HSBA", "HSBA.L"),
    ("AZN", "AZN.L"),
    ("ULVR", "ULVR.L"),
    ("SHEL", "SHEL.L"),
)


class _TerminalHTTP(Exception):
    """HTTP status that must not be retried (caller still sees HTTPError)."""

    def __init__(self, status: int) -> None:
        self.status = status


def _chart_url(symbol: str, start: datetime, end: datetime, events: str | None) -> str:
    url = CHART_URL.format(
        symbol=symbol,
        start=int(start.timestamp()),
        end=int(end.timestamp()),
    )
    if events is None:
        return url
    if not events or any(char.isspace() for char in events):
        raise ValueError("yahoo events must be a non-empty string without whitespace")
    return f"{url}&events={quote(events, safe='')}"


def fetch_yahoo_chart(
    symbol: str,
    *,
    start: datetime,
    end: datetime,
    timeout: float = 30.0,
    retries: int = 3,
    backoff_s: float = 2.0,
    events: str | None = None,
) -> dict[str, object]:
    """Download one Yahoo v8 daily chart.

    ``events=None`` keeps the historical quote URL. Research total-return
    callers pass ``div,split`` so the payload includes corporate actions.
    """
    url = _chart_url(symbol, start, end, events)

    def once() -> dict[str, object]:
        try:
            status, body = pooled_request(
                "GET",
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
                max_bytes=50_000_000,
            )
        except IoError as exc:
            raise urllib.error.URLError(str(exc)) from exc
        if status in (429, 500, 502, 503, 504):
            raise urllib.error.HTTPError(url, status, f"HTTP {status}", Message(), None)
        if status >= 400:
            raise _TerminalHTTP(status)
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("yahoo chart payload is not an object")
        return payload

    try:
        return call_with_retry(
            once,
            retries=retries,
            backoff_s=backoff_s,
            max_backoff_s=max(backoff_s, 8.0),
            jitter=True,
            retry_on=(urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError),
        )
    except _TerminalHTTP as exc:
        raise urllib.error.HTTPError(
            url, exc.status, f"HTTP {exc.status}", Message(), None
        ) from None


def parse_yahoo_chart(
    payload: dict[str, object], *, security_id: str, yahoo_symbol: str
) -> pl.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("yahoo chart payload must be an object")
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise ValueError("yahoo chart payload missing chart object")
    result = chart.get("result")
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("yahoo chart payload must contain exactly one result")
    block = result[0]
    if not isinstance(block, dict):
        raise ValueError("yahoo chart result must be an object")
    stamps = block.get("timestamp")
    indicators = block.get("indicators")
    if not isinstance(stamps, list):
        raise ValueError("yahoo chart timestamp must be a list")
    if not isinstance(indicators, dict):
        raise ValueError("yahoo chart indicators must be an object")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or len(quotes) != 1:
        raise ValueError("yahoo chart indicators must contain exactly one quote")
    quote = quotes[0]
    if not isinstance(quote, dict):
        raise ValueError("yahoo chart quote must be an object")
    columns: dict[str, list[Any]] = {}
    for field in ("open", "high", "low", "close", "volume"):
        values = quote.get(field)
        if not isinstance(values, list):
            raise ValueError(f"yahoo chart {field} must be a list")
        columns[field] = values
    n = len(stamps)
    if any(len(values) != n for values in columns.values()):
        raise ValueError("yahoo chart timestamp and OHLCV lists must have equal lengths")
    opens = columns["open"]
    highs = columns["high"]
    lows = columns["low"]
    closes = columns["close"]
    volumes = columns["volume"]
    ingested = datetime.now(tz=UTC)
    suffix = ".uk" if yahoo_symbol.endswith(".L") else ".us"
    rows: list[dict[str, object]] = []
    for i in range(n):
        try:
            day = datetime.fromtimestamp(int(stamps[i]), tz=UTC).date()
            ts = session_close(day, suffix)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise ValueError(f"yahoo chart timestamp at row {i} is invalid") from exc
        try:
            opn = float(opens[i])
            high = float(highs[i])
            low = float(lows[i])
            close = float(closes[i])
            volume = float(volumes[i])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"yahoo chart OHLCV at row {i} must be numeric") from exc
        if not all(np_finite(v) and v > 0 for v in (opn, high, low, close)):
            raise ValueError(f"yahoo chart OHLC at row {i} must be finite and positive")
        if volume < 0 or not np_finite(volume):
            raise ValueError(f"yahoo chart volume at row {i} must be finite and non-negative")
        if high < low or not (low <= opn <= high and low <= close <= high):
            raise ValueError(f"yahoo chart OHLC envelope at row {i} is invalid")
        rows.append(
            {
                "security_id": security_id,
                "symbol": security_id,
                "event_time": ts,
                "available_time": ts,
                "ingested_time": ingested,
                "source": SOURCE,
                "revision_id": REVISION,
                "open": opn,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "currency": "GBP" if suffix == ".uk" else "USD",
                "session": "rth",
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def collect_bars(
    names: tuple[tuple[str, str], ...],
    *,
    start: datetime | None,
    end: datetime | None,
    pause_s: float,
    max_workers: int,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Fetch, parse, and bound-check every name; return (bars, per-name errors)."""
    start_ts = start or datetime(2019, 1, 2, tzinfo=UTC)
    end_ts = end or datetime.now(tz=UTC)

    def _one(pair: tuple[str, str]) -> tuple[str, pl.DataFrame | None, str | None]:
        security_id, symbol = pair
        try:
            payload = fetch_yahoo_chart(symbol, start=start_ts, end=end_ts)
            frame = parse_yahoo_chart(payload, security_id=security_id, yahoo_symbol=symbol)
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            return security_id, None, str(exc)
        return security_id, frame, None

    fetched = map_ordered(
        _one,
        list(names),
        max_workers=max_workers,
        min_interval_s=pause_s,
    )
    frames: list[pl.DataFrame] = []
    errors: dict[str, str] = {}
    for security_id, frame, err in fetched:
        if err is not None:
            errors[security_id] = err
            continue
        if frame is not None and not frame.is_empty():
            frames.append(frame)
    if not frames:
        return pl.DataFrame(), errors
    return (
        pl.concat(frames, how="diagonal_relaxed").sort(["event_time", "security_id"]),
        errors,
    )


def download_yahoo_universe(
    root: Path,
    names: tuple[tuple[str, str], ...] = YAHOO_US,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    pause_s: float = 0.15,
    sectors: dict[str, str] | None = None,
    max_workers: int = 4,
) -> dict[str, object]:
    bars, errors = collect_bars(
        names, start=start, end=end, pause_s=pause_s, max_workers=max_workers
    )
    if bars.is_empty():
        return {
            "status": "empty",
            "n_names": 0,
            "errors": errors,
            "source": SOURCE,
            "pit_convention": "session_close_equals_available",
            "vendor_adjusted": True,
            "sip_vintage": False,
        }
    paths = write_file_lake(bars, root, sectors=sectors)
    return {
        "status": "ok",
        "n_names": int(bars["security_id"].n_unique()),
        "n_bars": int(bars.height),
        "paths": {k: str(v) for k, v in paths.items()},
        "errors": errors,
        "source": SOURCE,
        "revision_id": REVISION,
        "pit_convention": "session_close_equals_available",
        "vendor_adjusted": True,
        "sip_vintage": False,
        "sector_map": "static_current_classification" if sectors else "none",
        "champion_alias": False,
        "research_only": True,
    }
