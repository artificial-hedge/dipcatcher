"""Focused malformed-payload regressions for Yahoo and licensed-vendor ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from quant_fund.data.adapters.vendor_http import (
    DEFAULT_API_KEY_ENV,
    VendorDataError,
    VendorHttpAdapter,
    VendorHttpConfig,
)
from quant_fund.data.adapters.yahoo_eod import parse_yahoo_chart


def _yahoo_payload() -> dict[str, Any]:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [1_704_153_600, 1_704_240_000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [102.0, 103.0],
                                "low": [99.0, 100.0],
                                "close": [101.0, 102.0],
                                "volume": [1_000.0, 1_100.0],
                            }
                        ]
                    },
                }
            ]
        }
    }


def _parse_yahoo(payload: dict[str, Any]) -> object:
    return parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"chart": None},
        {"chart": {"result": None}},
        {"chart": {"result": []}},
        {"chart": {"result": [None]}},
        {"chart": {"result": [{}]}},
        {"chart": {"result": [{"timestamp": [], "indicators": None}]}},
        {"chart": {"result": [{"timestamp": [], "indicators": {"quote": []}}]}},
        {"chart": {"result": [{"timestamp": [], "indicators": {"quote": [None]}}]}},
        {
            "chart": {
                "result": [
                    {
                        "timestamp": [],
                        "indicators": {"quote": [{"open": []}]},
                    }
                ]
            }
        },
    ],
    ids=(
        "missing-chart",
        "chart-not-object",
        "result-not-list",
        "empty-result",
        "block-not-object",
        "missing-timestamp-and-indicators",
        "indicators-not-object",
        "empty-quote",
        "quote-not-object",
        "missing-ohlcv-columns",
    ),
)
def test_yahoo_malformed_container_shapes_raise_value_error(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _parse_yahoo(payload)


def test_yahoo_valid_empty_chart_returns_empty_frame() -> None:
    payload = _yahoo_payload()
    block = payload["chart"]["result"][0]
    block["timestamp"] = []
    quote = block["indicators"]["quote"][0]
    for field in ("open", "high", "low", "close", "volume"):
        quote[field] = []

    assert _parse_yahoo(payload).is_empty()


@pytest.mark.parametrize("container", ("result", "quote"))
def test_yahoo_multiple_result_or_quote_blocks_rejected(container: str) -> None:
    payload = _yahoo_payload()
    if container == "result":
        blocks = payload["chart"]["result"]
    else:
        blocks = payload["chart"]["result"][0]["indicators"]["quote"]
    blocks.append(blocks[0])

    with pytest.raises(ValueError, match="exactly one"):
        _parse_yahoo(payload)


def test_yahoo_unrepresentable_number_raises_value_error() -> None:
    payload = _yahoo_payload()
    payload["chart"]["result"][0]["indicators"]["quote"][0]["open"][0] = 10**1000

    with pytest.raises(ValueError, match="must be numeric"):
        _parse_yahoo(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("high", 98.0),
        ("low", 104.0),
        ("open", 104.0),
        ("open", 98.0),
        ("close", 104.0),
        ("close", 98.0),
    ],
)
def test_yahoo_broken_ohlc_envelope_is_not_repaired_or_dropped(field: str, value: float) -> None:
    payload = _yahoo_payload()
    payload["chart"]["result"][0]["indicators"]["quote"][0][field][0] = value

    with pytest.raises(ValueError, match="envelope"):
        _parse_yahoo(payload)


def _vendor_adapter_body(body: bytes) -> VendorHttpAdapter:
    return VendorHttpAdapter(
        VendorHttpConfig(
            vendor="error_test_vendor",
            base_url="https://vendor.example.test/v1",
            license_acknowledged=True,
            environ={DEFAULT_API_KEY_ENV: "test-key"},
        ),
        transport=lambda _url, _headers: body,
        clock=lambda: datetime(2024, 1, 5, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"null", "payload is not a JSON object"),
        (b"[]", "payload is not a JSON object"),
        (b"{}", "missing 'bars' list"),
        (b'{"bars": {}}', "missing 'bars' list"),
        (b'{"bars": [null]}', "row is not a JSON object"),
    ],
)
def test_vendor_malformed_payloads_raise_data_errors(body: bytes, message: str) -> None:
    adapter = _vendor_adapter_body(body)

    with pytest.raises(VendorDataError, match=message):
        adapter.get_bars()


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open", 102.0),
        ("open", 98.0),
        ("close", 102.0),
        ("close", 98.0),
    ],
)
def test_vendor_open_close_outside_envelope_rejected(field: str, value: float) -> None:
    import json

    row = _vendor_row()
    row[field] = value
    adapter = _vendor_adapter_body(json.dumps({"bars": [row]}).encode("utf-8"))

    with pytest.raises(VendorDataError, match="outside.*envelope"):
        adapter.get_bars()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("open", 10**1000, "must be numeric"),
        ("ts", "0001-01-01T00:00:00+05:00", "cannot be normalized to UTC"),
        ("ts", "9999-12-31T23:59:59-05:00", "cannot be normalized to UTC"),
    ],
)
def test_vendor_unrepresentable_values_raise_data_errors(
    field: str, value: Any, message: str
) -> None:
    import json

    row = _vendor_row()
    row[field] = value
    adapter = _vendor_adapter_body(json.dumps({"bars": [row]}).encode("utf-8"))

    with pytest.raises(VendorDataError, match=message):
        adapter.get_bars()
