"""Bounded parser fuzzing with synthetic Yahoo/vendor payloads; no network."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_fund.data.adapters.stooq import session_close
from quant_fund.data.adapters.vendor_http import (
    DEFAULT_API_KEY_ENV,
    VendorDataError,
    VendorHttpAdapter,
    VendorHttpConfig,
)
from quant_fund.data.adapters.yahoo_eod import parse_yahoo_chart
from tests.property._profiles import adversarial_settings

CLOCK_NOW = datetime(2024, 2, 1, tzinfo=UTC)
OHLCV_FIELDS = ("open", "high", "low", "close", "volume")
BAD_NUMBERS = (
    None,
    "not-a-number",
    [],
    {},
    float("nan"),
    float("inf"),
    -float("inf"),
    -1.0,
    10**1000,
)
BAD_TIMESTAMPS = (
    None,
    "not-a-time",
    "0001-01-01T00:00:00+05:00",
    "9999-12-31T23:59:59-05:00",
    [],
    {},
    float("nan"),
    float("inf"),
    10**100,
    -(10**100),
)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _yahoo_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [row["timestamp"] for row in rows],
                    "indicators": {
                        "quote": [{field: [row[field] for row in rows] for field in OHLCV_FIELDS}]
                    },
                }
            ]
        }
    }


def _yahoo_quote(payload: dict[str, Any]) -> dict[str, list[Any]]:
    return payload["chart"]["result"][0]["indicators"]["quote"][0]


def _yahoo_block(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["chart"]["result"][0]


@st.composite
def _yahoo_cases(
    draw: st.DrawFn,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    count = draw(st.integers(1, 5))
    first_stamp = draw(st.integers(946_684_800, 1_704_067_200))
    yahoo_symbol = draw(st.sampled_from(("AAPL", "BP.L")))
    security_id = "BP" if yahoo_symbol.endswith(".L") else "AAPL"
    rows = []
    for i in range(count):
        low = draw(st.floats(0.01, 100_000.0, allow_nan=False, allow_infinity=False))
        high = low + draw(st.floats(0.0, 1_000.0, allow_nan=False, allow_infinity=False))
        open_ = low + (high - low) * draw(st.floats(0.0, 1.0))
        close = low + (high - low) * draw(st.floats(0.0, 1.0))
        rows.append(
            {
                "timestamp": first_stamp + i * 86_400,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": draw(st.floats(0.0, 1e12, allow_nan=False, allow_infinity=False)),
            }
        )
    return _yahoo_payload(rows), rows, security_id, yahoo_symbol


def _vendor_row() -> dict[str, Any]:
    return {
        "security_id": "SEC_A",
        "ts": "2024-01-02T21:00:00Z",
        "release_ts": "2024-01-02T21:05:00Z",
        "ingest_ts": "2024-01-02T21:06:00Z",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000.0,
    }


@st.composite
def _vendor_cases(
    draw: st.DrawFn,
) -> tuple[dict[str, Any], dict[tuple[str, datetime], dict[str, Any]]]:
    count = draw(st.integers(1, 5))
    base = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    rows = []
    expected = {}
    for i in range(count):
        low = draw(st.floats(0.01, 100_000.0, allow_nan=False, allow_infinity=False))
        high = low + draw(st.floats(0.0, 1_000.0, allow_nan=False, allow_infinity=False))
        open_ = low + (high - low) * draw(st.floats(0.0, 1.0))
        close = low + (high - low) * draw(st.floats(0.0, 1.0))
        ts = base + timedelta(minutes=5 * i)
        security_id = draw(st.sampled_from(("SEC_A", "SEC_B", "SEC_C")))
        offset_hours = draw(st.sampled_from((-5, 0, 5)))
        ts_text = (
            _iso(ts)
            if offset_hours == 0
            else ts.astimezone(timezone(timedelta(hours=offset_hours))).isoformat()
        )
        row = {
            "security_id": security_id,
            "ts": ts_text,
            "release_ts": _iso(ts + timedelta(minutes=1)),
            "ingest_ts": _iso(ts + timedelta(minutes=2)),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": draw(st.floats(0.0, 1e12, allow_nan=False, allow_infinity=False)),
        }
        rows.append(row)
        expected[(security_id, ts)] = row
    return {"bars": rows}, expected


def _vendor_adapter_body(body: bytes) -> VendorHttpAdapter:
    return VendorHttpAdapter(
        VendorHttpConfig(
            vendor="fuzz_vendor",
            base_url="https://vendor.example.test/v1",
            license_acknowledged=True,
            environ={DEFAULT_API_KEY_ENV: "test-key"},
        ),
        transport=lambda _url, _headers: body,
        clock=lambda: CLOCK_NOW,
    )


def _vendor_adapter(payload: object) -> VendorHttpAdapter:
    return _vendor_adapter_body(json.dumps(payload).encode("utf-8"))


def _assert_stable_failure(call: Callable[[], object], error_type: type[Exception]) -> None:
    with pytest.raises(error_type) as first:
        call()
    with pytest.raises(error_type) as second:
        call()
    assert str(first.value) == str(second.value)


@given(case=_yahoo_cases())
@adversarial_settings()
def test_yahoo_valid_payloads_preserve_rows_and_invariants(
    case: tuple[dict[str, Any], list[dict[str, Any]], str, str],
) -> None:
    payload, rows, security_id, yahoo_symbol = case
    frame = parse_yahoo_chart(payload, security_id=security_id, yahoo_symbol=yahoo_symbol)
    suffix = ".uk" if yahoo_symbol.endswith(".L") else ".us"

    assert frame.height == len(rows)
    assert frame["security_id"].to_list() == [security_id] * len(rows)
    assert frame["symbol"].to_list() == [security_id] * len(rows)
    for actual, expected in zip(frame.to_dicts(), rows, strict=True):
        for field in OHLCV_FIELDS:
            assert actual[field] == float(expected[field])
        expected_ts = session_close(
            datetime.fromtimestamp(expected["timestamp"], tz=UTC).date(), suffix
        )
        assert actual["event_time"] == expected_ts
        assert actual["available_time"] == expected_ts
        assert actual["ingested_time"].utcoffset() == timedelta(0)
        assert actual["low"] <= actual["open"] <= actual["high"]
        assert actual["low"] <= actual["close"] <= actual["high"]
        assert actual["volume"] >= 0


@given(
    case=_yahoo_cases(),
    field=st.sampled_from(OHLCV_FIELDS),
    new_length=st.integers(0, 6),
)
@adversarial_settings()
def test_yahoo_mismatched_columns_fail_closed(
    case: tuple[dict[str, Any], list[dict[str, Any]], str, str],
    field: str,
    new_length: int,
) -> None:
    payload, rows, security_id, yahoo_symbol = case
    if new_length == len(rows):
        new_length += 1
    _yahoo_quote(payload)[field] = [1.0] * new_length

    _assert_stable_failure(
        lambda: parse_yahoo_chart(payload, security_id=security_id, yahoo_symbol=yahoo_symbol),
        ValueError,
    )


@given(
    case=_yahoo_cases(),
    field=st.sampled_from(OHLCV_FIELDS),
    index=st.integers(0, 4),
    bad_value=st.sampled_from(BAD_NUMBERS),
)
@adversarial_settings()
def test_yahoo_bad_numeric_cells_raise_stable_errors(
    case: tuple[dict[str, Any], list[dict[str, Any]], str, str],
    field: str,
    index: int,
    bad_value: Any,
) -> None:
    payload, rows, security_id, yahoo_symbol = case
    _yahoo_quote(payload)[field][index % len(rows)] = bad_value

    _assert_stable_failure(
        lambda: parse_yahoo_chart(payload, security_id=security_id, yahoo_symbol=yahoo_symbol),
        ValueError,
    )


@given(
    case=_yahoo_cases(),
    index=st.integers(0, 4),
    bad_value=st.sampled_from(BAD_TIMESTAMPS),
)
@adversarial_settings()
def test_yahoo_bad_timestamps_raise_stable_errors(
    case: tuple[dict[str, Any], list[dict[str, Any]], str, str],
    index: int,
    bad_value: Any,
) -> None:
    payload, rows, security_id, yahoo_symbol = case
    _yahoo_block(payload)["timestamp"][index % len(rows)] = bad_value

    _assert_stable_failure(
        lambda: parse_yahoo_chart(payload, security_id=security_id, yahoo_symbol=yahoo_symbol),
        ValueError,
    )


@given(case=_vendor_cases())
@adversarial_settings()
def test_vendor_valid_payloads_preserve_rows_and_security_identities(
    case: tuple[dict[str, Any], dict[tuple[str, datetime], dict[str, Any]]],
) -> None:
    payload, expected = case
    frame = _vendor_adapter(payload).get_bars()

    assert frame.height == len(payload["bars"])
    assert set(frame["security_id"]).issubset({key[0] for key in expected})
    for actual in frame.to_dicts():
        source = expected[(actual["security_id"], actual["event_time"])]
        for field in OHLCV_FIELDS:
            assert actual[field] == float(source[field])
        assert actual["event_time"].utcoffset() == timedelta(0)
        assert actual["release_ts"].utcoffset() == timedelta(0)
        assert actual["ingest_ts"].utcoffset() == timedelta(0)
        assert actual["available_time"] == actual["release_ts"]
        assert actual["ingested_time"] == actual["ingest_ts"]
        assert actual["event_time"] <= actual["ingest_ts"]
        assert actual["release_ts"] <= actual["ingest_ts"]
        assert actual["low"] <= actual["open"] <= actual["high"]
        assert actual["low"] <= actual["close"] <= actual["high"]
        assert actual["volume"] >= 0


@given(
    field=st.sampled_from(("ts", "release_ts", "ingest_ts")),
    bad_value=st.sampled_from(BAD_TIMESTAMPS + ("2024-01-02T21:00:00",)),
)
@adversarial_settings()
def test_vendor_bad_timestamps_raise_stable_errors(field: str, bad_value: Any) -> None:
    row = _vendor_row()
    row[field] = bad_value
    adapter = _vendor_adapter({"bars": [row]})

    _assert_stable_failure(adapter.get_bars, VendorDataError)


@given(field=st.sampled_from(OHLCV_FIELDS), bad_value=st.sampled_from(BAD_NUMBERS))
@adversarial_settings()
def test_vendor_bad_numeric_cells_raise_stable_errors(field: str, bad_value: Any) -> None:
    row = _vendor_row()
    row[field] = bad_value
    adapter = _vendor_adapter({"bars": [row]})

    _assert_stable_failure(adapter.get_bars, VendorDataError)


@given(suffix=st.binary(max_size=32))
@adversarial_settings()
def test_vendor_invalid_utf8_json_raises_stable_errors(suffix: bytes) -> None:
    adapter = _vendor_adapter_body(b"\xff" + suffix)

    _assert_stable_failure(adapter.get_bars, VendorDataError)
