"""StooqSource registry wiring + fetch contract."""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.data.adapters import stooq as stooq_mod
from quant_fund.data.adapters import yahoo_eod as yahoo_mod
from quant_fund.data.collector import collect_source
from quant_fund.data.sources.adapters import StooqSource, YahooSource
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.registry import get_source, source_names

_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-02,100,101,99,100.5,1000\n"
    "2024-01-03,100.5,102,100,101.7,1200\n"
    "2024-01-04,101.7,103,101,102.1,900\n"
)


def _stub_csv(monkeypatch: pytest.MonkeyPatch, fail: set[str] | None = None) -> None:
    fail = fail or set()

    def fake(symbol: str, **_: object) -> str:
        if symbol in fail:
            raise OSError("boom")
        return _CSV

    monkeypatch.setattr(stooq_mod, "fetch_stooq_csv", fake)


def test_registry_resolves_stooq_eod_and_alias() -> None:
    assert "stooq" in source_names()
    assert get_source("stooq").name == "stooq"


def test_fetch_returns_pit_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_csv(monkeypatch)
    frame = StooqSource().fetch(names="AAPL:aapl.us,MSFT:msft.us", pause_s=0)
    assert frame.height == 6
    assert set(frame["security_id"].unique()) == {"AAPL", "MSFT"}
    for column in ("event_time", "available_time", "ingested_time", "source", "revision_id"):
        assert column in frame.columns
    assert frame["source"].unique().to_list() == ["stooq"]
    assert frame["revision_id"].unique().to_list() == [stooq_mod.REVISION]
    # Session-close PIT convention: event == available, both aware UTC.
    assert frame.filter(pl.col("event_time") != pl.col("available_time")).is_empty()


def test_fetch_start_end_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_csv(monkeypatch)
    frame = StooqSource().fetch(
        names="AAPL:aapl.us", start="2024-01-03", end="2024-01-03", pause_s=0
    )
    assert frame.height == 1
    assert frame["event_time"][0] == datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def test_fetch_strict_fails_on_any_name_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_csv(monkeypatch, fail={"msft.us"})
    with pytest.raises(SourceError, match="1/2 names"):
        StooqSource().fetch(names="AAPL:aapl.us,MSFT:msft.us", pause_s=0)


def test_fetch_non_strict_keeps_survivors(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_csv(monkeypatch, fail={"msft.us"})
    frame = StooqSource().fetch(names="AAPL:aapl.us,MSFT:msft.us", strict="false", pause_s=0)
    assert set(frame["security_id"].unique()) == {"AAPL"}


def test_fetch_empty_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_csv(monkeypatch, fail={"aapl.us"})
    with pytest.raises(SourceError, match="no bars"):
        StooqSource().fetch(names="AAPL:aapl.us", strict=False, pause_s=0)


def test_fetch_bad_names_and_strict_values_raise() -> None:
    source = StooqSource()
    with pytest.raises(ValueError, match="vendor_symbol"):
        source.fetch(names="aapl.us")  # missing the ':' separator
    with pytest.raises(ValueError, match="strict"):
        source.fetch(names="AAPL:aapl.us", strict="maybe")


def test_collect_source_end_to_end_stooq(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_csv(monkeypatch)
    result = collect_source("stooq", tmp_path, fetch_kwargs={"names": "AAPL:aapl.us", "pause_s": 0})
    assert result.data == tmp_path / "raw" / "sources" / "stooq.parquet"
    assert result.receipt.exists()
    assert result.frame.height == 3


_YAHOO_PAYLOAD: dict[str, object] = {
    "chart": {
        "result": [
            {
                "timestamp": [1704153600, 1704240000],  # 2024-01-02, 2024-01-03
                "indicators": {
                    "quote": [
                        {
                            "open": [100.0, 100.5],
                            "high": [101.0, 102.0],
                            "low": [99.0, 100.0],
                            "close": [100.5, 101.7],
                            "volume": [1000, 1200],
                        }
                    ]
                },
            }
        ]
    }
}


def _stub_yahoo(monkeypatch: pytest.MonkeyPatch, fail: set[str] | None = None) -> None:
    fail = fail or set()

    def fake(symbol: str, **_: object) -> dict[str, object]:
        if symbol in fail:
            raise OSError("boom")
        return _YAHOO_PAYLOAD

    monkeypatch.setattr(yahoo_mod, "fetch_yahoo_chart", fake)


def test_registry_resolves_yahoo() -> None:
    assert "yahoo" in source_names()
    assert get_source("yahoo").name == "yahoo"


def test_yahoo_fetch_pit_frame_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_yahoo(monkeypatch)
    frame = YahooSource().fetch(names="AAPL:AAPL", start="2024-01-01", pause_s=0)
    assert frame.height == 2
    assert frame["source"].unique().to_list() == ["yahoo"]
    assert frame["revision_id"].unique().to_list() == [yahoo_mod.REVISION]
    assert frame.filter(pl.col("event_time") != pl.col("available_time")).is_empty()


def test_yahoo_fetch_strict_and_empty_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_yahoo(monkeypatch, fail={"MSFT"})
    with pytest.raises(SourceError, match="yahoo fetch failed"):
        YahooSource().fetch(names="AAPL:AAPL,MSFT:MSFT", pause_s=0)
    with pytest.raises(SourceError, match="no bars"):
        YahooSource().fetch(names="MSFT:MSFT", pause_s=0, strict=False)


def _yahoo_payload_with_arrays(**overrides: object) -> dict[str, object]:
    arrays: dict[str, object] = {
        "timestamp": [1704153600, 1704240000],
        "open": [100.0, 100.5],
        "high": [101.0, 102.0],
        "low": [99.0, 100.0],
        "close": [100.5, 101.7],
        "volume": [1000, 1200],
    }
    arrays.update(overrides)
    return {
        "chart": {
            "result": [
                {
                    "timestamp": arrays["timestamp"],
                    "indicators": {
                        "quote": [
                            {
                                "open": arrays["open"],
                                "high": arrays["high"],
                                "low": arrays["low"],
                                "close": arrays["close"],
                                "volume": arrays["volume"],
                            }
                        ]
                    },
                }
            ]
        }
    }


def test_stooq_rejects_future_session_close() -> None:
    future_csv = _CSV.replace("2024-01-02", "2999-01-02")
    with pytest.raises(ValueError, match="stooq session close.*after ingested_time"):
        stooq_mod.parse_stooq_csv(future_csv, security_id="AAPL", stooq_symbol="aapl.us")


def test_stooq_rejects_duplicate_source_dates() -> None:
    duplicate_csv = _CSV.replace("2024-01-03", "2024-01-02")
    with pytest.raises(ValueError, match="stooq duplicate derived key"):
        stooq_mod.parse_stooq_csv(duplicate_csv, security_id="AAPL", stooq_symbol="aapl.us")


def test_yahoo_rejects_future_session_close() -> None:
    future = int(datetime(2999, 1, 2, tzinfo=UTC).timestamp())
    payload = _yahoo_payload_with_arrays(
        timestamp=[future],
        open=[100.0],
        high=[101.0],
        low=[99.0],
        close=[100.5],
        volume=[1000],
    )
    with pytest.raises(ValueError, match="yahoo session close.*after ingested_time"):
        yahoo_mod.parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")


def test_yahoo_rejects_timestamps_mapping_to_one_session() -> None:
    payload = _yahoo_payload_with_arrays(timestamp=[1704153600, 1704157200])
    with pytest.raises(ValueError, match="yahoo duplicate derived key"):
        yahoo_mod.parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")


@pytest.mark.parametrize("field", ["timestamp", "open", "high", "low", "close", "volume"])
def test_yahoo_rejects_non_list_arrays(field: str) -> None:
    payload = _yahoo_payload_with_arrays(**{field: 100})
    with pytest.raises(ValueError, match=rf"yahoo {field} must be a list"):
        yahoo_mod.parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")


def test_yahoo_rejects_unequal_array_lengths() -> None:
    payload = _yahoo_payload_with_arrays(volume=[1000])
    with pytest.raises(ValueError, match="yahoo chart arrays must have equal lengths"):
        yahoo_mod.parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")


def test_yahoo_equal_arrays_preserve_per_row_null_handling() -> None:
    payload = _yahoo_payload_with_arrays(
        timestamp=[1704153600, 1704240000, 1704326400],
        open=[100.0, None, 102.0],
        high=[101.0, 102.0, 103.0],
        low=[99.0, 100.0, 101.0],
        close=[100.5, 101.7, 102.5],
        volume=[1000, 1200, None],
    )
    frame = yahoo_mod.parse_yahoo_chart(payload, security_id="AAPL", yahoo_symbol="AAPL")
    assert frame.height == 2
    assert frame["volume"].to_list() == [1000.0, 0.0]


def test_yahoo_collect_filters_out_of_range_session_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _yahoo_payload_with_arrays(
        timestamp=[1704153600, 1704240000, 1704326400],
        open=[100.0, 100.5, 101.7],
        high=[101.0, 102.0, 103.0],
        low=[99.0, 100.0, 101.0],
        close=[100.5, 101.7, 102.1],
        volume=[1000, 1200, 900],
    )

    def fake(*_: object, **__: object) -> dict[str, object]:
        return payload

    monkeypatch.setattr(yahoo_mod, "fetch_yahoo_chart", fake)
    frame = YahooSource().fetch(names="AAPL:AAPL", start="2024-01-03", end="2024-01-03", pause_s=0)
    assert frame.height == 1
    assert frame["event_time"][0] == datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def test_mixed_case_uk_suffixes_use_london_close_and_gbp() -> None:
    stooq_frame = stooq_mod.parse_stooq_csv(_CSV, security_id="VOD", stooq_symbol="vod.UK")
    yahoo_frame = yahoo_mod.parse_yahoo_chart(
        _YAHOO_PAYLOAD, security_id="VOD", yahoo_symbol="VOD.l"
    )
    for frame in (stooq_frame, yahoo_frame):
        assert frame["currency"].unique().to_list() == ["GBP"]
        assert frame["event_time"][0] == datetime(2024, 1, 2, 16, 30, tzinfo=UTC)
