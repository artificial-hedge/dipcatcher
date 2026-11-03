"""Tests for the venue-carry fetch scripts' pure parsing helpers.

Covers the four public-archive/venue fetchers that had no test coverage at
all:

- ``scripts/fetch_binance_vision_carry.py`` — S3 key/month discovery, epoch
  unit sniffing, kline/funding CSV parsing, zip extraction.
- ``scripts/fetch_binance_daily_tail.py`` — daily-archive date windowing and
  the public funding-history pagination loop.
- ``scripts/fetch_dydx_carry.py`` — indexer universe filtering and the
  candle/funding frame builders.
- ``scripts/fetch_hyperliquid_carry.py`` — ``/info`` universe filtering and
  the candle/funding frame builders.
- ``scripts/fetch_okx_carry.py`` — instrument universe ranking and the
  candle/funding frame builders.

Every network call is faked. Nothing here reaches S3, the dYdX indexer,
Hyperliquid, or OKX, and no data directory is written.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    """Import a script module by its canonical name (scripts/ is on sys.path)."""
    import importlib

    return importlib.import_module(name)


bv = _load("fetch_binance_vision_carry")
daily = _load("fetch_binance_daily_tail")
dydx = _load("fetch_dydx_carry")
hl = _load("fetch_hyperliquid_carry")
okx = _load("fetch_okx_carry")


# --------------------------------------------------------------------------- #
# _months_from_keys
# --------------------------------------------------------------------------- #


def test_months_from_keys_extracts_ym_tokens() -> None:
    keys = [
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01.zip",
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-11.zip",
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2025-03.zip",
    ]
    assert bv._months_from_keys(keys) == ["2024-11", "2025-01", "2025-03"]


def test_months_from_keys_ignores_non_month_keys() -> None:
    keys = [
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01.zip",
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-01.zip",
        "data/futures/um/monthly/klines/BTCUSDT/1d/README.txt",
        "data/futures/um/monthly/klines/BTCUSDT/1d/2025-01.zip",
    ]
    assert bv._months_from_keys(keys) == ["2025-01"]


def test_months_from_keys_rejects_non_numeric_year() -> None:
    assert bv._months_from_keys(["data/x/abcd-01.zip"]) == []


def test_months_from_keys_deduplicates_and_sorts() -> None:
    keys = [
        "data/x/BTCUSDT-1d-2025-02.zip",
        "data/x/BTCUSDT-1d-2025-02.zip",
        "data/x/BTCUSDT-1d-2025-01.zip",
    ]
    assert bv._months_from_keys(keys) == ["2025-01", "2025-02"]


def test_months_from_keys_empty() -> None:
    assert bv._months_from_keys([]) == []


# --------------------------------------------------------------------------- #
# _epoch_to_dt
# --------------------------------------------------------------------------- #


def test_epoch_to_dt_sniffs_microseconds() -> None:
    # 2025-01-01T00:00:00Z expressed in microseconds (16 digits).
    ts = 1_735_689_600_000_000
    assert bv._epoch_to_dt(ts) == datetime(2025, 1, 1, tzinfo=UTC)


def test_epoch_to_dt_sniffs_milliseconds() -> None:
    ts = 1_735_689_600_000
    assert bv._epoch_to_dt(ts) == datetime(2025, 1, 1, tzinfo=UTC)


def test_epoch_to_dt_micro_and_milli_agree() -> None:
    assert bv._epoch_to_dt(1_735_689_600_000_000) == bv._epoch_to_dt(1_735_689_600_000)


def test_epoch_to_dt_is_utc_aware() -> None:
    assert bv._epoch_to_dt(1_735_689_600_000).tzinfo is not None


# --------------------------------------------------------------------------- #
# _parse_kline_csv
# --------------------------------------------------------------------------- #


def test_parse_kline_csv_skips_header_and_parses_row() -> None:
    raw = (
        b"open_time,open,high,low,close,volume,close_time\n"
        b"1735689600000,1.5,2.5,0.5,2.0,100.0,1735775999999\n"
    )
    rows = bv._parse_kline_csv(raw, "BTC", "binance")
    assert len(rows) == 1
    row = rows[0]
    assert row["security_id"] == "BTC"
    assert row["source"] == "binance"
    assert row["open"] == 1.5
    assert row["high"] == 2.5
    assert row["low"] == 0.5
    assert row["close"] == 2.0
    assert row["volume"] == 100.0
    assert row["event_time"] == datetime(2025, 1, 1, tzinfo=UTC)


def test_parse_kline_csv_skips_short_rows() -> None:
    raw = b"1735689600000,1.5,2.5,0.5,2.0\n"
    assert bv._parse_kline_csv(raw, "BTC", "binance") == []


def test_parse_kline_csv_skips_non_numeric_timestamp() -> None:
    raw = b"open_time,open,high,low,close,volume\nnotanumber,1,2,3,4,5\n"
    assert bv._parse_kline_csv(raw, "BTC", "binance") == []


def test_parse_kline_csv_skips_unparsable_floats() -> None:
    raw = b"1735689600000,abc,2.5,0.5,2.0,100.0\n"
    assert bv._parse_kline_csv(raw, "BTC", "binance") == []


def test_parse_kline_csv_empty_input() -> None:
    assert bv._parse_kline_csv(b"", "BTC", "binance") == []


def test_parse_kline_csv_accepts_microsecond_epochs() -> None:
    raw = b"1735689600000000,1.5,2.5,0.5,2.0,100.0\n"
    rows = bv._parse_kline_csv(raw, "BTC", "binance")
    assert rows[0]["event_time"] == datetime(2025, 1, 1, tzinfo=UTC)


def test_parse_kline_csv_keeps_order_and_multiple_rows() -> None:
    raw = b"1735689600000,1,2,3,4,5\n1735776000000,1,2,3,4,5\n"
    rows = bv._parse_kline_csv(raw, "BTC", "binance")
    assert [r["event_time"] for r in rows] == [
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    ]


# --------------------------------------------------------------------------- #
# _parse_funding_csv
# --------------------------------------------------------------------------- #


def test_parse_funding_csv_reads_third_column() -> None:
    raw = b"calc_time,funding_interval_hours,last_funding_rate\n1735689600000,8,0.0001\n"
    rows = bv._parse_funding_csv(raw, "BTC")
    assert len(rows) == 1
    assert rows[0]["security_id"] == "BTC"
    assert rows[0]["value"] == 0.0001
    assert rows[0]["event_time"] == datetime(2025, 1, 1, tzinfo=UTC)


def test_parse_funding_csv_skips_short_rows() -> None:
    assert bv._parse_funding_csv(b"1735689600000,8\n", "BTC") == []


def test_parse_funding_csv_skips_header() -> None:
    assert (
        bv._parse_funding_csv(b"calc_time,funding_interval_hours,last_funding_rate\n", "BTC") == []
    )


def test_parse_funding_csv_skips_unparsable_rate() -> None:
    assert bv._parse_funding_csv(b"1735689600000,8,abc\n", "BTC") == []


def test_parse_funding_csv_negative_rate_preserved() -> None:
    rows = bv._parse_funding_csv(b"1735689600000,8,-0.0002\n", "BTC")
    assert rows[0]["value"] == -0.0002


# --------------------------------------------------------------------------- #
# _dl_zip_csv
# --------------------------------------------------------------------------- #


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return buf.getvalue()


def test_dl_zip_csv_extracts_first_csv_member(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = _zip_bytes({"BTCUSDT-1d-2025-01.csv": b"a,b\n"})
    monkeypatch.setattr(bv, "_http", lambda url, **kw: blob)
    assert bv._dl_zip_csv("https://example/x.zip") == b"a,b\n"


def test_dl_zip_csv_ignores_non_csv_members(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = _zip_bytes({"README.txt": b"nope"})
    monkeypatch.setattr(bv, "_http", lambda url, **kw: blob)
    assert bv._dl_zip_csv("https://example/x.zip") is None


def test_dl_zip_csv_returns_none_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(url: str, **kw):
        raise FileNotFoundError(url)

    monkeypatch.setattr(bv, "_http", _boom)
    assert bv._dl_zip_csv("https://example/missing.zip") is None


def test_dl_zip_csv_returns_none_on_bad_zip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bv, "_http", lambda url, **kw: b"not a zip file")
    assert bv._dl_zip_csv("https://example/bad.zip") is None


def test_dl_zip_csv_requests_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_http(url: str, **kw):
        seen["binary"] = kw.get("binary")
        return _zip_bytes({"x.csv": b"1\n"})

    monkeypatch.setattr(bv, "_http", _fake_http)
    bv._dl_zip_csv("https://example/x.zip")
    assert seen["binary"] is True


# --------------------------------------------------------------------------- #
# list_keys
# --------------------------------------------------------------------------- #

_S3_NS = "http://s3.amazonaws.com/doc/2006-03-01/"


def _s3_xml(keys: list[str], truncated: bool) -> str:
    contents = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<ListBucketResult xmlns="{_S3_NS}">'
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
        f"{contents}"
        f"</ListBucketResult>"
    )


def test_list_keys_returns_single_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bv, "_http", lambda url: _s3_xml(["a.zip", "b.zip"], truncated=False))
    assert bv.list_keys("data/x/") == ["a.zip", "b.zip"]


def test_list_keys_paginates_until_truncation_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        _s3_xml(["a.zip"], truncated=True),
        _s3_xml(["b.zip"], truncated=False),
    ]
    seen: list[str] = []

    def _fake_http(url: str):
        seen.append(url)
        return pages[len(seen) - 1]

    monkeypatch.setattr(bv, "_http", _fake_http)
    assert bv.list_keys("data/x/") == ["a.zip", "b.zip"]
    assert len(seen) == 2
    assert "marker=a.zip" in seen[1]


def test_list_keys_stops_on_empty_page_even_if_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _fake_http(url: str):
        calls["n"] += 1
        return _s3_xml([], truncated=True)

    monkeypatch.setattr(bv, "_http", _fake_http)
    assert bv.list_keys("data/x/") == []
    assert calls["n"] == 1


def test_list_keys_builds_prefixed_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(bv, "_http", lambda url: seen.append(url) or _s3_xml([], truncated=False))
    bv.list_keys("data/spot/")
    assert seen[0].startswith(bv.S3)
    assert "prefix=data/spot/" in seen[0]
    assert "max-keys=1000" in seen[0]


# --------------------------------------------------------------------------- #
# fetch_binance_daily_tail._days_for
# --------------------------------------------------------------------------- #


def test_days_for_keeps_dates_in_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        daily,
        "list_keys",
        lambda prefix, marker=None: [
            "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-01.zip",
            "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-02.zip",
            "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-03.zip",
        ],
    )
    out = daily._days_for("BTC", "spot", date(2025, 1, 2))
    assert out == [date(2025, 1, 2), date(2025, 1, 3)]


def test_days_for_drops_unparsable_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        daily,
        "list_keys",
        lambda prefix, marker=None: [
            "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-01.zip",
            "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-not-a-date.zip",
        ],
    )
    out = daily._days_for("BTC", "spot", date(2025, 1, 1))
    assert out == [date(2025, 1, 1)]


def test_days_for_drops_dates_after_today(monkeypatch: pytest.MonkeyPatch) -> None:
    future = (daily.TODAY.replace(year=daily.TODAY.year + 1)).isoformat()
    monkeypatch.setattr(
        daily,
        "list_keys",
        lambda prefix, marker=None: [f"data/x/BTCUSDT-1d-{future}.zip"],
    )
    assert daily._days_for("BTC", "spot", date(2020, 1, 1)) == []


def test_days_for_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-01.zip"
    monkeypatch.setattr(daily, "list_keys", lambda prefix, marker=None: [key, key])
    assert daily._days_for("BTC", "spot", date(2025, 1, 1)) == [date(2025, 1, 1)]


def test_days_for_passes_marker_at_start(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_list_keys(prefix, marker=None):
        seen["prefix"] = prefix
        seen["marker"] = marker
        return []

    monkeypatch.setattr(daily, "list_keys", _fake_list_keys)
    daily._days_for("BTC", "spot", date(2025, 1, 1))
    assert seen["prefix"] == "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-"
    assert seen["marker"] == "data/spot/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2025-01-01"


def test_fetch_coin_tail_parses_downloaded_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily, "_days_for", lambda coin, kind, start: [date(2025, 1, 1)])
    monkeypatch.setattr(
        daily,
        "_dl_zip_csv",
        lambda url: b"1735689600000,1,2,3,4,5\n",
    )
    rows = daily.fetch_coin_tail("BTC", "spot", date(2025, 1, 1))
    assert len(rows) == 1
    assert rows[0]["source"] == "binance_daily"
    assert rows[0]["security_id"] == "BTC"


def test_fetch_coin_tail_skips_missing_archives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daily, "_days_for", lambda coin, kind, start: [date(2025, 1, 1)])
    monkeypatch.setattr(daily, "_dl_zip_csv", lambda url: None)
    assert daily.fetch_coin_tail("BTC", "spot", date(2025, 1, 1)) == []


# --------------------------------------------------------------------------- #
# fetch_binance_daily_tail._fapi_funding
# --------------------------------------------------------------------------- #


def test_fapi_funding_returns_single_page(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = [{"fundingTime": 1, "fundingRate": "0.0001"}]

    def _fake_read(req, *, timeout):
        del req, timeout
        return json.dumps(payload).encode()

    monkeypatch.setattr(daily, "read_vendor", _fake_read)
    assert daily._fapi_funding("BTC", 0, 10) == payload


def test_fapi_funding_paginates_on_full_page(monkeypatch: pytest.MonkeyPatch) -> None:
    full = [{"fundingTime": i, "fundingRate": "0.0"} for i in range(1000)]
    tail = [{"fundingTime": 1000, "fundingRate": "0.0"}]
    pages = [full, tail]
    calls = {"n": 0}

    def _fake_read(req, *, timeout):
        del req, timeout
        data = pages[min(calls["n"], len(pages) - 1)]
        calls["n"] += 1
        return json.dumps(data).encode()

    monkeypatch.setattr(daily, "read_vendor", _fake_read)
    rows = daily._fapi_funding("BTC", 0, 10_000)
    assert len(rows) == 1001
    assert calls["n"] == 2


def test_fapi_funding_stops_on_empty_page(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_read(req, *, timeout):
        del req, timeout
        return json.dumps([]).encode()

    monkeypatch.setattr(daily, "read_vendor", _fake_read)
    assert daily._fapi_funding("BTC", 0, 10) == []


def test_fapi_funding_raises_after_exhausted_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _fake_read(req, *, timeout):
        del req, timeout
        calls["n"] += 1
        raise urllib.error.URLError("down")

    monkeypatch.setattr(daily, "read_vendor", _fake_read)
    monkeypatch.setattr(daily.time, "sleep", lambda _s: None)
    with pytest.raises(urllib.error.URLError):
        daily._fapi_funding("BTC", 0, 10)
    assert calls["n"] == 6


# --------------------------------------------------------------------------- #
# fetch_dydx_carry
# --------------------------------------------------------------------------- #


def test_dydx_list_universe_filters_active_usd_markets() -> None:
    payload = {
        "markets": {
            "BTC-USD": {"status": "ACTIVE"},
            "ETH-USD": {"status": "ACTIVE"},
            "SOL-USD": {"status": "OFFLINE"},
            "ABC-USDC": {"status": "ACTIVE"},
        }
    }
    monkey = pytest.MonkeyPatch()
    monkey.setattr(dydx, "_get", lambda path, **kw: payload)
    try:
        assert dydx.list_universe() == ["BTC", "ETH"]
    finally:
        monkey.undo()


def test_dydx_bars_frame_maps_candles() -> None:
    candles = [
        {
            "startedAt": "2025-01-01T00:00:00.000Z",
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
            "baseTokenVolume": "10",
        }
    ]
    frame = dydx._bars_frame("BTC", candles, "dydx")
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["security_id"] == "BTC"
    assert row["source"] == "dydx"
    assert row["open"] == 1.0
    assert row["close"] == 1.5
    assert row["volume"] == 10.0
    assert row["event_time"] == datetime(2025, 1, 1, tzinfo=UTC)


def test_dydx_bars_frame_empty_has_schema() -> None:
    frame = dydx._bars_frame("BTC", [], "dydx")
    assert frame.height == 0
    assert frame.schema["event_time"] == pl.Datetime("us", UTC)
    assert frame.schema["security_id"] == pl.String


def test_dydx_bars_frame_dedupes_on_event_time_and_sid() -> None:
    candles = [
        {
            "startedAt": "2025-01-01T00:00:00.000Z",
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
            "baseTokenVolume": "10",
        }
    ] * 2
    assert dydx._bars_frame("BTC", candles, "dydx").height == 1


def test_dydx_fund_frame_maps_events() -> None:
    events = [{"effectiveAt": "2025-01-01T00:00:00.000Z", "rate": "0.0001"}]
    frame = dydx._fund_frame("BTC", events)
    assert frame.height == 1
    assert frame.row(0, named=True)["value"] == 0.0001


def test_dydx_fund_frame_empty_has_schema() -> None:
    frame = dydx._fund_frame("BTC", [])
    assert frame.height == 0
    assert frame.schema["value"] == pl.Float64


def test_dydx_one_uses_ticker_prefix_as_coin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dydx, "fetch_candles", lambda ticker, *a, **k: [])
    monkeypatch.setattr(dydx, "fetch_funding", lambda ticker, *a, **k: [])
    coin, perp, fund = dydx._one("BTC-USD")
    assert coin == "BTC"
    assert perp.height == 0
    assert fund.height == 0


def test_dydx_fetch_candles_paginates_backwards(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        {"candles": [{"startedAt": "2025-01-02T00:00:00.000Z"}] * 2},
        {"candles": [{"startedAt": "2025-01-01T00:00:00.000Z"}]},
    ]
    calls = {"n": 0}

    def _fake_get(path, **kw):
        data = pages[min(calls["n"], len(pages) - 1)]
        calls["n"] += 1
        return data

    monkeypatch.setattr(dydx, "_get", _fake_get)
    rows = dydx.fetch_candles("BTC-USD", limit=2)
    assert len(rows) == 3
    assert calls["n"] == 2


def test_dydx_fetch_funding_paginates_backwards(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        {"historicalFunding": [{"effectiveAt": "2025-01-02T00:00:00.000Z"}] * 2},
        {"historicalFunding": [{"effectiveAt": "2025-01-01T00:00:00.000Z"}]},
    ]
    calls = {"n": 0}

    def _fake_get(path, **kw):
        data = pages[min(calls["n"], len(pages) - 1)]
        calls["n"] += 1
        return data

    monkeypatch.setattr(dydx, "_get", _fake_get)
    rows = dydx.fetch_funding("BTC-USD", limit=2)
    assert len(rows) == 3
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# fetch_hyperliquid_carry
# --------------------------------------------------------------------------- #


def test_hl_list_perp_universe_skips_delisted_and_namespaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "universe": [
            {"name": "BTC", "isDelisted": False},
            {"name": "ETH", "isDelisted": True},
            {"name": "xyz:ABC", "isDelisted": False},
        ]
    }
    monkeypatch.setattr(hl, "_post", lambda body, **kw: payload)
    assert hl.list_perp_universe() == ["BTC"]


def test_hl_list_spot_pairs_maps_base_to_universe_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "tokens": [{"index": 0, "name": "PURR"}, {"index": 1, "name": "USDC"}],
        "universe": [
            {"name": "PURR/USDC", "tokens": [0, 1]},
            {"name": "PURR/USDC-B", "tokens": [0]},
        ],
    }
    monkeypatch.setattr(hl, "_post", lambda body, **kw: payload)
    assert hl.list_spot_pairs() == {"PURR": "PURR/USDC"}


def test_hl_bars_frame_maps_candles() -> None:
    candles = [{"t": 1_735_689_600_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "9"}]
    frame = hl._bars_frame("BTC", candles, "hyperliquid")
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["security_id"] == "BTC"
    assert row["source"] == "hyperliquid"
    assert row["close"] == 1.5
    assert row["event_time"] == datetime(2025, 1, 1, tzinfo=UTC)


def test_hl_bars_frame_empty_has_schema() -> None:
    frame = hl._bars_frame("BTC", [], "hyperliquid")
    assert frame.height == 0
    assert frame.schema["event_time"] == pl.Datetime("us", UTC)


def test_hl_fund_frame_maps_events() -> None:
    events = [{"time": 1_735_689_600_000, "fundingRate": "0.0002"}]
    frame = hl._fund_frame("BTC", events)
    assert frame.height == 1
    assert frame.row(0, named=True)["value"] == 0.0002


def test_hl_fund_frame_empty_has_schema() -> None:
    frame = hl._fund_frame("BTC", [])
    assert frame.height == 0
    assert frame.schema["value"] == pl.Float64


def test_hl_one_skips_spot_when_not_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def _fake_candles(coin: str, interval: str = "1d"):
        seen.append((coin, interval))
        return []

    monkeypatch.setattr(hl, "fetch_candles", _fake_candles)
    monkeypatch.setattr(hl, "fetch_funding", lambda coin: [])
    coin, perp, spot, fund = hl._one("BTC", "PURR/USDC", "4h", want_spot=False)
    assert coin == "BTC"
    assert perp.height == 0
    assert spot.height == 0
    assert fund.height == 0
    assert seen == [("BTC", "4h")]


def test_hl_one_uses_spot_name_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def _fake_candles(coin: str, interval: str = "1d"):
        seen.append(coin)
        return []

    monkeypatch.setattr(hl, "fetch_candles", _fake_candles)
    monkeypatch.setattr(hl, "fetch_funding", lambda coin: [])
    hl._one("BTC", "PURR/USDC")
    assert seen == ["BTC", "PURR/USDC"]


def test_hl_sample_funding_sums_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hl, "_post", lambda body, **kw: [{"fundingRate": "0.0001"}, {"fundingRate": "-0.0002"}]
    )
    assert hl.sample_funding("BTC", 0, 1) == pytest.approx(-0.0001)


def test_hl_prescreen_keeps_material_funding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hl,
        "sample_funding",
        lambda coin, start, end: 0.002 if coin == "BTC" else 0.0,
    )
    assert hl.prescreen(["BTC", "ETH"], thresh=0.0008) == ["BTC"]


# --------------------------------------------------------------------------- #
# fetch_okx_carry
# --------------------------------------------------------------------------- #


def test_okx_list_universe_ranks_by_dollar_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/api/v5/public/instruments?instType=SWAP": [
            {"instId": "BTC-USDT-SWAP", "state": "live"},
            {"instId": "ETH-USDT-SWAP", "state": "live"},
            {"instId": "DEAD-USDT-SWAP", "state": "suspend"},
        ],
        "/api/v5/market/tickers?instType=SWAP": [
            {"instId": "BTC-USDT-SWAP", "volCcy24h": "100", "last": "10"},
            {"instId": "ETH-USDT-SWAP", "volCcy24h": "1000", "last": "10"},
            {"instId": "DEAD-USDT-SWAP", "volCcy24h": "99999", "last": "10"},
        ],
        "/api/v5/public/instruments?instType=SPOT": [
            {"instId": "BTC-USDT", "state": "live"},
            {"instId": "ETH-USDT", "state": "live"},
        ],
    }

    def _fake_get(path, **kw):
        return {"code": "0", "data": responses[path]}

    monkeypatch.setattr(okx, "_get", _fake_get)
    assert okx.list_universe(2) == ["ETH", "BTC"]


def test_okx_list_universe_requires_live_spot_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/api/v5/public/instruments?instType=SWAP": [
            {"instId": "BTC-USDT-SWAP", "state": "live"},
        ],
        "/api/v5/market/tickers?instType=SWAP": [
            {"instId": "BTC-USDT-SWAP", "volCcy24h": "100", "last": "10"},
        ],
        "/api/v5/public/instruments?instType=SPOT": [],
    }

    def _fake_get(path, **kw):
        return {"code": "0", "data": responses[path]}

    monkeypatch.setattr(okx, "_get", _fake_get)
    assert okx.list_universe(10) == []


def test_okx_get_raises_on_error_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(okx.time, "sleep", lambda _s: None)

    def _fake_read(req, *, timeout):
        del req, timeout
        return json.dumps({"code": "50011", "msg": "boom"}).encode()

    monkeypatch.setattr(okx, "read_vendor", _fake_read)
    with pytest.raises(RuntimeError, match="okx error"):
        okx._get("/api/v5/public/instruments?instType=SWAP")


def test_okx_bars_frame_maps_candles() -> None:
    candles = [["1735689600000", "1", "2", "0.5", "1.5", "10", "15", "12", "1"]]
    frame = okx._bars_frame("BTC", candles, "okx")
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["security_id"] == "BTC"
    assert row["source"] == "okx"
    assert row["close"] == 1.5
    # volume comes from volCcyQuote (index 7) when present
    assert row["volume"] == 12.0
    assert row["event_time"] == datetime(2025, 1, 1, tzinfo=UTC)


def test_okx_bars_frame_falls_back_to_base_volume() -> None:
    candles = [["1735689600000", "1", "2", "0.5", "1.5", "10", "15", "", "1"]]
    frame = okx._bars_frame("BTC", candles, "okx")
    assert frame.row(0, named=True)["volume"] == 10.0


def test_okx_fund_frame_maps_events() -> None:
    events = [{"fundingTime": "1735689600000", "fundingRate": "0.0003"}]
    frame = okx._fund_frame("BTC", events)
    assert frame.height == 1
    assert frame.row(0, named=True)["value"] == 0.0003


def test_okx_fund_frame_empty_has_schema() -> None:
    frame = okx._fund_frame("BTC", [])
    assert frame.height == 0
    assert frame.schema["value"] == pl.Float64


def test_okx_bars_frame_empty_raises_column_not_found() -> None:
    """OKX has no empty-frame schema guard, unlike dYdX / Hyperliquid.

    ``_bars_frame`` builds ``pl.DataFrame(recs)`` unconditionally, so an empty
    candle list yields a zero-column frame and the ``unique([...])`` call
    raises. The dYdX and Hyperliquid equivalents return an explicitly-typed
    empty frame instead. Pinned here so the asymmetry is visible rather than
    a surprise at runtime.
    """
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        okx._bars_frame("BTC", [], "okx")


def test_okx_one_uses_suffixed_instrument_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    candle = ["1735689600000", "1", "2", "3", "4", "5", "6", "7", "1"]
    monkeypatch.setattr(
        okx, "fetch_candles", lambda inst_id, *a, **k: seen.append(inst_id) or [candle]
    )
    monkeypatch.setattr(okx, "fetch_funding", lambda inst_id, *a, **k: [])
    coin, perp, spot, fund = okx._one("BTC")
    assert coin == "BTC"
    assert seen == ["BTC-USDT-SWAP", "BTC-USDT"]
    assert perp.height == 1
    assert spot.height == 1
    assert fund.height == 0


def test_okx_fetch_candles_paginates_via_after(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        [["1735689600000", "1", "2", "3", "4", "5", "6", "7", "1"]] * 2,
        [["1735603200000", "1", "2", "3", "4", "5", "6", "7", "1"]],
    ]
    calls = {"n": 0}

    def _fake_get(path, **kw):
        data = pages[min(calls["n"], len(pages) - 1)]
        calls["n"] += 1
        return {"code": "0", "data": data}

    monkeypatch.setattr(okx, "_get", _fake_get)
    rows = okx.fetch_candles("BTC-USDT-SWAP", limit=2)
    assert len(rows) == 3
    assert calls["n"] == 2
