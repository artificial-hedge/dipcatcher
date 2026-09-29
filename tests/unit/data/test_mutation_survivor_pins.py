"""Mutation-audit survivor pins — data path.

Hand-applied mutants at these lines passed the module's existing test slice
green; each test below encodes the intended contract and kills the mutant.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.config.models import UniverseConfig
from quant_fund.data.ingest import PublicMarketProvider, ingest
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.data.sources import base
from quant_fund.data.sources.adapters import BinancePublicDataSource
from quant_fund.data.sources.base import HttpClient, SourceAdapter, SourceError
from quant_fund.data.universe import (
    _apply_vectorized_tickers,
    _attach_static_master,
    build_membership_panel,
    membership_asof,
)
from quant_fund.schemas.errors import PointInTimeError

ingest_mod = importlib.import_module("quant_fund.data.ingest")

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _provider_with_stub() -> tuple[PublicMarketProvider, pl.DataFrame]:
    cfg = load_config("configs/paper.yaml")
    cfg.data.source = "binance_public_data"
    provider = PublicMarketProvider(cfg)
    bars = pl.DataFrame(
        {
            "security_id": ["A", "B"] * 3,
            "event_time": [T0 + timedelta(days=i) for i in range(3) for _ in range(2)],
            "available_time": [T0 + timedelta(days=i) for i in range(3) for _ in range(2)],
            "ingested_time": [T0 + timedelta(days=i) for i in range(3) for _ in range(2)],
            "open": [10.0] * 6,
            "high": [11.0] * 6,
            "low": [9.0] * 6,
            "close": [10.0] * 6,
            "volume": [1e5] * 6,
            "source": ["binance_public_data"] * 6,
            "revision_id": ["1d"] * 6,
        }
    ).sort("security_id", "event_time")

    class _StubAdapter(SourceAdapter):
        def get_bars(self, **kwargs: Any) -> pl.DataFrame:
            return bars

    provider.adapter = _StubAdapter()
    return provider, bars


def test_get_bars_window_bounds_are_inclusive() -> None:
    """start/end filter bars inclusively — the boundary bar stays."""
    provider, _ = _provider_with_stub()
    t1 = T0 + timedelta(days=1)
    out = provider.get_bars(start=t1)
    assert min(out["event_time"].to_list()) == t1
    out = provider.get_bars(end=t1)
    assert max(out["event_time"].to_list()) == t1


def test_get_bars_security_ids_keep_only_requested() -> None:
    """security_ids is an allow-list, not a deny-list."""
    provider, _ = _provider_with_stub()
    out = provider.get_bars(security_ids=["A"])
    assert set(out["security_id"].to_list()) == {"A"}


def test_ingest_applies_corporate_actions_to_silver(tmp_path, monkeypatch) -> None:
    """Non-empty actions must reach adjust_prices — splits change silver prices."""
    d0, d1 = T0, T0 + timedelta(days=1)
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [d0, d1],
            "open": [100.0, 50.0],
            "high": [100.0, 50.0],
            "low": [100.0, 50.0],
            "close": [100.0, 50.0],
            "volume": [1e5, 2e5],
        }
    )
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [d1],
            "action_type": ["split"],
            "factor": [2.0],
        }
    )
    provider = SimpleNamespace(
        get_bars=lambda *a, **k: bars,
        get_corporate_actions=lambda *a, **k: actions,
        get_security_master=lambda *a, **k: pl.DataFrame(),
    )
    monkeypatch.setattr(ingest_mod, "make_provider", lambda cfg: provider)
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.data.source = "synthetic"
    paths = ingest(cfg)
    silver = pl.read_parquet(paths["silver"])
    adjusted = silver.filter(pl.col("event_time") == d0)["close_split_adjusted"][0]
    assert adjusted == pytest.approx(50.0)


def test_validate_feature_frame_tolerates_old_availability() -> None:
    """assert_pit_safe(max_available, decision) — args must not be swapped."""
    frame = pl.DataFrame(
        {
            "available_time": [T0],
            "ingested_time": [T0],
            "x": [1.0],
        }
    )
    validate_feature_frame(frame, T0 + timedelta(days=7))


def test_validate_feature_frame_prefers_max_source_availability() -> None:
    """The conservative per-source max governs — a late restatement rejects."""
    frame = pl.DataFrame(
        {
            "available_time": [T0],
            "max_source_available_time": [T0 + timedelta(days=30)],
            "ingested_time": [T0],
            "x": [1.0],
        }
    )
    with pytest.raises(PointInTimeError):
        validate_feature_frame(frame, T0 + timedelta(days=7))


def _uni_bars(n_a: int = 25, n_c: int = 5) -> pl.DataFrame:
    def rows(sid: str, n: int) -> dict[str, list[Any]]:
        return {
            "security_id": [sid] * n,
            "event_time": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)],
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.0] * n,
            "volume": [2e5] * n,
        }

    a = pl.DataFrame(rows("A", n_a))
    c = pl.DataFrame(rows("C", n_c))
    return pl.concat([a, c])


def _uni_master() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": ["A", "C", "F"],
            "exchange": ["XNYS", "XNYS", "XNYS"],
            "sector": ["t", "t", "t"],
            "industry": ["i", "i", "i"],
            "security_type": ["common_stock", "common_stock", "common_stock"],
            "ticker": ["AAA", "CCC", "FFF"],
        }
    )


def _uni_cfg(min_adv: float = 1.0) -> UniverseConfig:
    return UniverseConfig(
        min_price=1.0,
        min_adv=min_adv,
        min_history_bars=1,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )


def test_membership_excludes_null_adv() -> None:
    """Null ADV fills with 0.0 — a too-short history must not leapfrog min_adv."""
    mem = membership_asof(_uni_bars(), _uni_master(), datetime(2020, 2, 1, tzinfo=UTC), _uni_cfg())
    assert "C" not in mem["security_id"].to_list()
    assert "A" in mem["security_id"].to_list()


def test_ticker_change_never_applies_future_names() -> None:
    """join_asof stays backward: a later ticker_change cannot rename history."""
    panel = pl.DataFrame(
        {
            "security_id": ["A"],
            "asof": [datetime(2020, 1, 5, tzinfo=UTC)],
            "ticker": ["OLD"],
        }
    )
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 10, tzinfo=UTC)],
            "action_type": ["ticker_change"],
            "new_ticker": ["NEW"],
        }
    )
    out = _apply_vectorized_tickers(panel, actions)
    assert out["ticker"].to_list() == ["OLD"]


def test_membership_snap_never_pulls_future_bars() -> None:
    """The vectorized as-of join snaps backward — a mid-day as-of must see the
    prior bar, never the next day's."""
    closes = [10.0] * 25
    closes[22] = 99.0  # Jan 23: the first bar strictly after the as-of
    a = pl.DataFrame(
        {
            "security_id": ["A"] * 25,
            "event_time": [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(25)],
            "open": [10.0] * 25,
            "high": [11.0] * 25,
            "low": [9.0] * 25,
            "close": closes,
            "volume": [2e5] * 25,
        }
    )
    panel = build_membership_panel(
        a,
        _uni_master(),
        [datetime(2020, 1, 22, 12, tzinfo=UTC)],
        _uni_cfg(),
    )
    assert panel["px"].to_list() == [10.0]
    assert panel["event_time"].to_list() == [datetime(2020, 1, 22, tzinfo=UTC)]


def test_static_master_hidden_once_validity_ends() -> None:
    """A master row valid only *up to* the as-of is not visible at the as-of."""
    asof = datetime(2020, 1, 5, tzinfo=UTC)
    panel = pl.DataFrame({"security_id": ["A"], "asof": [asof]})
    master = pl.DataFrame(
        {
            "security_id": ["A"],
            "sector": ["t"],
            "valid_to": [asof],
        }
    )
    out = _attach_static_master(panel, master)
    assert out["sector"].to_list() == [None]


def test_klines_with_start_time_paginate_not_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """start_time forces the paginated branch — startTime reaches the wire."""
    calls: list[str] = []

    def fake_request(method: str, url: str, **kw: Any) -> tuple[int, bytes]:
        calls.append(url)
        if "startTime=1000" in url:
            return 200, b'[[1000, "1", "2", "0.5", "1.5", "10", 1001]]'
        return 200, b"[]"

    monkeypatch.setattr(base, "pooled_request", fake_request)
    src = BinancePublicDataSource(client=HttpClient(retries=0))
    src.fetch(symbol="BTCUSDT", interval="1d", start_time=1000, limit=2)
    assert calls and "startTime=1000" in calls[0]


def test_http_client_rejects_4xx_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """4xx is not success — only >= 400 statuses raise."""
    monkeypatch.setattr(base, "pooled_request", lambda *a, **k: (404, b'{"err": "not found"}'))
    client = HttpClient(retries=0)
    with pytest.raises(SourceError):
        client.get_json("https://example.invalid/x")
