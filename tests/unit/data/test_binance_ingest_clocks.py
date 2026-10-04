"""SYNTHETIC REST responses: ingest/gold clock compatibility, not market evidence."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from quant_fund.config.models import AppConfig
from quant_fund.data.ingest import _close_label_binance_klines, ingest
from quant_fund.data.sources.adapters import BinanceMarketSource, BinancePublicDataSource
from quant_fund.data.sources.base import HttpClient
from quant_fund.pipeline.dataset import build_gold, panel
from quant_fund.schemas.errors import PointInTimeError


def _payload(step_ms: int) -> list[list[object]]:
    start = 1_577_836_800_000
    return [
        [
            start + i * step_ms,
            "10",
            "12",
            "9",
            str(10 + i / 100),
            "100000",
            start + (i + 1) * step_ms - 1,
        ]
        for i in range(80)
    ]


def _config(root: Path, source: str, interval: str) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "source": source,
                "root": root,
                "source_interval": interval,
                "benchmark_id": "BTCUSDT",
            },
            "universe": {"min_history_bars": 20, "min_adv": 0},
        }
    )


@pytest.mark.parametrize("source", ["binance_public_data", "binance_market_websocket"])
@pytest.mark.parametrize(
    "interval,step_ms", [("1m", 60_000), ("1h", 3_600_000), ("1d", 86_400_000)]
)
def test_rest_ingest_gold_uses_exact_vendor_close(monkeypatch, tmp_path, source, interval, step_ms):
    payload = _payload(step_ms)
    calls = []

    def fake_json(self, url, **kwargs):
        calls.append(url)
        return payload

    monkeypatch.setattr(HttpClient, "get_json", fake_json)
    cfg = _config(tmp_path, source, interval)
    paths = ingest(cfg)
    bronze = pl.read_parquet(paths["bars"])
    silver = pl.read_parquet(paths["silver"])
    membership = pl.read_parquet(paths["universe"])
    expected_close = datetime.fromtimestamp(payload[0][6] / 1000, UTC)
    assert bronze["event_time"][0] == datetime(2020, 1, 1, tzinfo=UTC)
    assert silver["event_time"][0] == expected_close
    assert expected_close.microsecond == 999000
    assert_frame_equal(
        bronze.select("event_time"), silver.select(pl.col("bar_open_time").alias("event_time"))
    )
    assert_frame_equal(bronze.drop("event_time"), silver.select(bronze.columns).drop("event_time"))
    assert (silver["event_time"] == silver["bar_close_time"]).all()
    assert membership["asof"].min() >= silver["event_time"][19]
    assert set(membership["asof"]) <= set(silver["event_time"])
    feats, labels = build_gold(cfg)
    joined = panel(cfg)
    assert feats.height and labels.height and joined.height
    assert (feats["decision_time"] == feats["bar_close_time"]).all()
    assert (feats["max_source_available_time"] <= feats["decision_time"]).all()
    assert set(feats["event_time"]) == set(labels["event_time"]) == set(membership["asof"])
    assert len(calls) == 1  # downstream calls use persisted silver, no provider calls


def test_monthly_close_is_vendor_timestamp_not_fixed_duration(monkeypatch):
    opening = datetime(2020, 2, 1, tzinfo=UTC)
    closing = datetime(2020, 3, 1, tzinfo=UTC) - timedelta(milliseconds=1)
    payload = [
        [
            int(opening.timestamp() * 1000),
            "10",
            "12",
            "9",
            "11",
            "100",
            int(closing.timestamp() * 1000),
        ]
    ]
    monkeypatch.setattr(HttpClient, "get_json", lambda *a, **k: payload)
    raw = BinancePublicDataSource().fetch(interval="1M")
    assert raw["event_time"][0] == opening
    assert _close_label_binance_klines(raw)["event_time"][0] == closing


def test_late_publication_still_fails_gold(monkeypatch, tmp_path):
    monkeypatch.setattr(HttpClient, "get_json", lambda *a, **k: _payload(86_400_000))
    original = BinancePublicDataSource.fetch

    def late_fetch(self, **kwargs):
        return original(self, **kwargs).with_columns(
            (pl.col("available_time") + pl.duration(hours=1)).alias("available_time")
        )

    monkeypatch.setattr(BinancePublicDataSource, "fetch", late_fetch)
    cfg = _config(tmp_path, "binance_public_data", "1d")
    ingest(cfg)
    with pytest.raises(PointInTimeError, match="future"):
        build_gold(cfg)


@pytest.mark.parametrize("mutation", ["missing", "null", "inverted", "duplicate", "mismatch"])
def test_malformed_close_provenance_fails_closed(monkeypatch, mutation):
    monkeypatch.setattr(HttpClient, "get_json", lambda *a, **k: _payload(86_400_000))
    bars = BinancePublicDataSource().fetch()
    if mutation == "missing":
        bars = bars.drop("bar_close_time")
    elif mutation == "null":
        bars = bars.with_columns(
            pl.lit(None, dtype=bars.schema["bar_close_time"]).alias("bar_close_time")
        )
    elif mutation == "inverted":
        bars = bars.with_columns(pl.col("bar_open_time").alias("bar_close_time"))
    elif mutation == "duplicate":
        bars = pl.concat([bars, bars.tail(1)])
    else:
        bars = bars.with_columns(
            (pl.col("event_time") + pl.duration(seconds=1)).alias("event_time")
        )
    with pytest.raises(PointInTimeError):
        _close_label_binance_klines(bars)


def test_websocket_trade_keeps_trade_and_publication_clocks():
    trade = BinanceMarketSource.normalize_trade(
        {
            "e": "trade",
            "s": "BTCUSDT",
            "p": "10",
            "q": "2",
            "T": 1_577_836_800_000,
            "E": 1_577_836_800_100,
        }
    )
    assert trade["available_time"][0] - trade["event_time"][0] == timedelta(milliseconds=100)
    assert "bar_close_time" not in trade.columns


def test_existing_open_labeled_silver_requires_explicit_refresh(monkeypatch, tmp_path):
    monkeypatch.setattr(HttpClient, "get_json", lambda *a, **k: _payload(86_400_000))
    cfg = _config(tmp_path, "binance_public_data", "1d")
    paths = ingest(cfg)
    old = pl.read_parquet(paths["silver"]).with_columns(pl.col("bar_open_time").alias("event_time"))
    old.write_parquet(paths["silver"])
    original_bytes = paths["silver"].read_bytes()
    with pytest.raises(PointInTimeError, match="future"):
        build_gold(cfg)
    assert paths["silver"].read_bytes() == original_bytes
    feats, _ = build_gold(cfg, refresh=True)
    assert (feats["event_time"] == feats["bar_close_time"]).all()
