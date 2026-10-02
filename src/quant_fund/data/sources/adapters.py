"""Concrete collectors for the requested public/open data source families."""

from __future__ import annotations

import calendar
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.data.sources.base import (
    SourceAdapter,
    SourceError,
    parse_time,
    pit_frame,
    query_url,
    utc_now,
)
from quant_fund.data.sources.normalize import csv_rows, normalize_observations, normalize_ohlcv

# Binance USDⓈ-M futures launched September 2019; no perp kline predates this.
PERP_EARLIEST_MS = 1567296000000  # 2019-09-01T00:00:00Z
# Binance spot trading launched July 2017; no spot kline predates this.
SPOT_EARLIEST_MS = 1498867200000  # 2017-07-01T00:00:00Z

_BINANCE_INTERVALS = frozenset(
    {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
)


def _require_kline_rows(payload: Any) -> list[list[Any]]:
    """Validate the kline row shape and timestamp fields before use."""
    if not isinstance(payload, list):
        raise SourceError("Binance klines response must be a list")
    for item in payload:
        if not (isinstance(item, list) and len(item) >= 7):
            raise SourceError("malformed Binance kline row")
        try:
            int(item[0])
            int(item[6])
        except (TypeError, ValueError) as exc:
            raise SourceError("malformed Binance kline row") from exc
    return payload


def _paginated_klines(
    client: Any,
    endpoint: str,
    *,
    symbol: str | None = None,
    pair: str | None = None,
    contract_type: str | None = None,
    interval: str,
    start_time: int | None,
    end_time: int | None,
    limit: int,
    max_pages: int,
    pause_seconds: float,
    earliest_ms: int = 0,
) -> list[list[Any]]:
    """Fetch every kline page for [start_time, end_time) in ascending order.

    Binance klines return at most ``limit`` rows per call starting at
    ``startTime``; without a startTime the API returns only the *most recent*
    bars, so full-history callers must paginate forward explicitly. The cursor
    advances to ``last_open_time + 1`` each page, which guarantees progress even
    when a page is truncated mid-interval. ``max_pages`` bounds total requests.
    ``earliest_ms`` is the caller's listing floor used when ``start_time`` is
    ``None`` — it differs between spot and futures products and defaults to
    the epoch so unspecified floors never silently truncate history.
    """
    if interval not in _BINANCE_INTERVALS:
        raise ValueError(f"unsupported Binance interval {interval!r}")
    if not 1 <= limit <= 1500:
        raise ValueError("Binance limit must be between 1 and 1500")
    if max_pages < 1:
        raise ValueError("max_pages must be >= 1")
    if pause_seconds < 0:
        raise ValueError("pause_seconds must be non-negative")
    cursor = int(start_time) if start_time is not None else earliest_ms
    rows: list[list[Any]] = []
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "interval": interval,
            "startTime": cursor,
            "endTime": end_time,
            "limit": limit,
        }
        if pair is not None:
            params["pair"] = pair.upper()
            params["contractType"] = contract_type
        else:
            params["symbol"] = str(symbol).upper()
        payload = client.get_json(query_url(endpoint, params))
        payload = _require_kline_rows(payload)
        if not payload:
            break
        rows.extend(payload)
        next_cursor = int(payload[-1][0]) + 1
        if len(payload) < limit or (end_time is not None and next_cursor >= int(end_time)):
            break
        cursor = next_cursor
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return rows


class BinancePublicDataSource(SourceAdapter):
    name = "binance_public_data"
    endpoint = "https://api.binance.com/api/v3/klines"

    def fetch(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
        max_pages: int = 1,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("Binance limit must be between 1 and 1000")
        if start_time is None and max_pages <= 1:
            # Legacy single-page semantics: no startTime -> most recent bars.
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {
                        "symbol": symbol.upper(),
                        "interval": interval,
                        "startTime": None,
                        "endTime": end_time,
                        "limit": limit,
                    },
                )
            )
        else:
            payload = _paginated_klines(
                self.client,
                self.endpoint,
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                max_pages=max_pages,
                pause_seconds=pause_seconds,
                earliest_ms=SPOT_EARLIEST_MS,
            )
        payload = _require_kline_rows(payload)
        now_ms = int(utc_now().timestamp() * 1000)
        rows = [
            {
                "security_id": symbol.upper(),
                "event_time": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                # Binance kline close time is the earliest conservative availability.
                "available_time": item[6],
            }
            for item in payload
            # Drop the still-open bar: its close/availability time is in the
            # future and its OHLCV values would keep mutating.
            if int(item[6]) <= now_ms
        ]
        if not rows:
            raise SourceError("Binance returned only an in-progress kline")
        # Preserve explicit vendor clocks independently of publication timing.
        # Direct adapters/bronze retain the vendor open label; the market ingest
        # boundary uses these fields to build a close-labeled silver panel.
        return normalize_ohlcv(rows, source=self.name, revision_id=interval).with_columns(
            pl.col("event_time").alias("bar_open_time"),
            pl.col("available_time").alias("bar_close_time"),
        )


class BinanceMarketSource(BinancePublicDataSource):
    """REST history plus pure normalization for Binance public WebSocket events."""

    name = "binance_market_websocket"

    @staticmethod
    def normalize_trade(message: dict[str, Any]) -> pl.DataFrame:
        data = message.get("data", message)
        if not isinstance(data, dict) or data.get("e") not in {"trade", "aggTrade"}:
            raise SourceError("unsupported Binance trade message")
        price = data.get("p")
        quantity = data.get("q")
        row = {
            "security_id": data.get("s"),
            "event_time": data.get("T", data.get("E")),
            "available_time": data.get("E", data.get("T")),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": quantity,
        }
        return normalize_ohlcv(
            [row], source=BinanceMarketSource.name, revision_id=str(data.get("a", "stream"))
        )


class BinanceUsdtmPerpSource(SourceAdapter):
    """Binance USDⓈ-M perpetual klines with full-history pagination.

    Unlike the spot adapter (single ≤1000-bar page), ``fetch`` pages forward
    from ``start_time`` (default: futures launch epoch) so multi-year hourly
    histories arrive complete. The still-open bar is dropped exactly like the
    spot source — its close time lies in the future and its OHLCV mutates.
    """

    name = "binance_usdtm_perp"
    endpoint = "https://fapi.binance.com/fapi/v1/klines"

    def fetch(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
        max_pages: int = 200,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        now_ms = int(utc_now().timestamp() * 1000)
        payload = _paginated_klines(
            self.client,
            self.endpoint,
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            max_pages=max_pages,
            pause_seconds=pause_seconds,
            earliest_ms=PERP_EARLIEST_MS,
        )
        rows = [
            {
                "security_id": symbol.upper(),
                "event_time": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "available_time": item[6],
            }
            for item in payload
            if int(item[6]) <= now_ms
        ]
        if not rows:
            raise SourceError("Binance perp returned only an in-progress kline")
        return normalize_ohlcv(rows, source=self.name, revision_id=f"{interval}.usdtm")


class BinanceFundingRateSource(SourceAdapter):
    """Binance USDⓈ-M funding-rate history (charged every 8h or 4h per symbol).

    The realized rate is only fixed at ``fundingTime``, so both event_time and
    available_time are stamped there — never earlier. Rows carry ``value`` =
    funding rate (decimal) and ``mark_price`` for auditability.
    """

    name = "binance_funding_rate"
    endpoint = "https://fapi.binance.com/fapi/v1/fundingRate"

    def fetch(
        self,
        *,
        symbol: str = "BTCUSDT",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
        max_pages: int = 200,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        if not 1 <= limit <= 1000:
            raise ValueError("Binance funding limit must be between 1 and 1000")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        cursor = int(start_time) if start_time is not None else PERP_EARLIEST_MS
        now_ms = int(utc_now().timestamp() * 1000)
        normalized_rows: list[dict[str, Any]] = []
        for _ in range(max_pages):
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {
                        "symbol": symbol.upper(),
                        "startTime": cursor,
                        "endTime": end_time,
                        "limit": limit,
                    },
                )
            )
            if not isinstance(payload, list):
                raise SourceError("Binance funding response must be a list")
            if not payload:
                break
            for item in payload:
                if not isinstance(item, dict):
                    raise SourceError("malformed Binance funding row")
                try:
                    funding_time = int(item["fundingTime"])
                    rate = float(item["fundingRate"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise SourceError(f"malformed Binance funding row: {item!r}") from exc
                if funding_time > now_ms:
                    # Not yet charged — the rate is only realized at fundingTime.
                    continue
                if not (rate == rate and abs(rate) < 10.0):
                    raise SourceError("funding rate is not finite")
                normalized_rows.append(
                    {
                        "security_id": str(item.get("symbol", symbol)).upper(),
                        "event_time": item["fundingTime"],
                        "available_time": item["fundingTime"],
                        "value": rate,
                        "raw_value": json.dumps(item, sort_keys=True),
                    }
                )
            next_cursor = int(payload[-1]["fundingTime"]) + 1
            if len(payload) < limit or (end_time is not None and next_cursor >= int(end_time)):
                break
            cursor = next_cursor
            if pause_seconds > 0:
                time.sleep(pause_seconds)
        if not normalized_rows:
            raise SourceError("Binance funding returned no rows")
        return normalize_observations(normalized_rows, source=self.name, revision_id="v1")


class BinancePerpUniverseSource(SourceAdapter):
    """Current USDⓈ-M perpetual universe: listing dates + 24h quote-volume rank.

    The public API only exposes *currently traded* symbols — this snapshot is a
    survivorship-truncated universe and is stamped ``available_time = now`` so
    downstream PIT logic can never pretend it was known earlier.
    """

    name = "binance_perp_universe"
    info_endpoint = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    ticker_endpoint = "https://fapi.binance.com/fapi/v1/ticker/24hr"

    def fetch(
        self,
        *,
        quote_asset: str = "USDT",
        top_n: int | None = None,
        min_quote_volume: float = 0.0,
    ) -> pl.DataFrame:
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be >= 1")
        info = self.client.get_json(self.info_endpoint)
        tickers = self.client.get_json(self.ticker_endpoint)
        symbols = info.get("symbols") if isinstance(info, dict) else None
        if not isinstance(symbols, list):
            raise SourceError("Binance exchangeInfo response has no symbols list")
        if not isinstance(tickers, list):
            raise SourceError("Binance 24hr ticker response must be a list")
        quote = quote_asset.upper()
        quote_volume = {}
        for item in tickers:
            if isinstance(item, dict) and "symbol" in item:
                try:
                    quote_volume[str(item["symbol"]).upper()] = float(item.get("quoteVolume", 0.0))
                except (TypeError, ValueError):
                    continue
        now = utc_now()
        rows: list[dict[str, Any]] = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            if (
                item.get("contractType") != "PERPETUAL"
                or str(item.get("quoteAsset", "")).upper() != quote
                or item.get("status") != "TRADING"
            ):
                continue
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            try:
                onboard_ms = int(item["onboardDate"])
            except KeyError as exc:
                raise SourceError(
                    f"perp universe member {symbol!r} is missing onboardDate"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise SourceError(
                    f"perp universe member {symbol!r} has invalid onboardDate"
                ) from exc
            volume = quote_volume.get(symbol)
            if volume is None or volume < float(min_quote_volume):
                continue
            rows.append(
                {
                    "security_id": symbol,
                    "event_time": onboard_ms,
                    "available_time": now,
                    "value": volume,
                    "base_asset": str(item.get("baseAsset", "")),
                    "onboard_ms": onboard_ms,
                }
            )
        rows.sort(key=lambda r: (-float(r["value"]), r["security_id"]))
        if top_n is not None:
            rows = rows[:top_n]
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        if not rows:
            raise SourceError("Binance perp universe resolved to zero symbols")
        return pit_frame(rows, source=self.name, revision_id="v1").sort(["rank"])


# Binance COIN-M delivery futures launched late August 2020; nothing predates this.
DELIVERY_EARLIEST_MS = 1595721600000  # 2020-07-26T00:00:00Z (first listed quarterlies)
# contractType values the continuousKlines endpoint accepts for spliced series.
DELIVERY_CONTRACT_TYPES = frozenset({"PERPETUAL", "CURRENT_QUARTERLY", "NEXT_QUARTERLY"})


def _require_delivery_symbol(symbol: str) -> str:
    """Named COIN-M contracts look like ``BTCUSD_250926`` (pair_deliveryYYMMDD)."""
    token = symbol.upper()
    left, sep, right = token.rpartition("_")
    if (
        not sep
        or not left
        or not left.replace("_", "").isalnum()
        or not (len(right) == 6 and right.isdigit())
    ):
        raise ValueError(f"delivery contract symbols look like 'BTCUSD_250926', got {symbol!r}")
    return token


class BinanceDeliveryKlinesSource(SourceAdapter):
    """Binance COIN-M delivery-futures klines for one named contract.

    ``symbol`` is the deliverable contract (``BTCUSD_250926``): the series is
    complete for that contract alone — it lists and expires, it does not roll.
    Full-history pagination forward from ``start_time`` (default: the COIN-M
    listing floor). The still-open bar is dropped, same as spot/perp — its
    close time lies in the future and its OHLCV mutates.
    """

    name = "binance_delivery_klines"
    endpoint = "https://dapi.binance.com/dapi/v1/klines"

    def fetch(
        self,
        *,
        symbol: str = "BTCUSD_QUARTERLY",
        interval: str = "1d",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
        max_pages: int = 200,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        contract = _require_delivery_symbol(symbol)
        now_ms = int(utc_now().timestamp() * 1000)
        payload = _paginated_klines(
            self.client,
            self.endpoint,
            symbol=contract,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            max_pages=max_pages,
            pause_seconds=pause_seconds,
            earliest_ms=DELIVERY_EARLIEST_MS,
        )
        rows = [
            {
                "security_id": contract,
                "event_time": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "available_time": item[6],
            }
            for item in payload
            if int(item[6]) <= now_ms
        ]
        if not rows:
            raise SourceError("Binance delivery returned only an in-progress kline")
        return normalize_ohlcv(rows, source=self.name, revision_id=f"{interval}.coinm")


class BinanceDeliveryContinuousSource(SourceAdapter):
    """Binance COIN-M continuous klines spliced over successive quarterlies.

    The ``continuousKlines`` endpoint stitches the front ``contractType``
    series (``CURRENT_QUARTERLY`` or ``NEXT_QUARTERLY``), so rows cross
    contract boundaries — the series rolls at delivery without an explicit
    roll flag from Binance. Rows carry ``contract_type`` so downstream code
    can never mistake the splice for one instrument; treat step changes near
    published delivery dates as roll artifacts, not price moves.
    """

    name = "binance_delivery_continuous"
    endpoint = "https://dapi.binance.com/dapi/v1/continuousKlines"

    def fetch(
        self,
        *,
        pair: str = "BTCUSD",
        contract_type: str = "CURRENT_QUARTERLY",
        interval: str = "1d",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1500,
        max_pages: int = 200,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        if contract_type.upper() not in DELIVERY_CONTRACT_TYPES:
            raise ValueError(f"contract_type must be one of {sorted(DELIVERY_CONTRACT_TYPES)}")
        if not pair or not pair.upper().isalnum():
            raise ValueError(f"malformed Binance pair {pair!r}")
        pair = pair.upper()
        contract_type = contract_type.upper()
        now_ms = int(utc_now().timestamp() * 1000)
        payload = _paginated_klines(
            self.client,
            self.endpoint,
            pair=pair,
            contract_type=contract_type,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            max_pages=max_pages,
            pause_seconds=pause_seconds,
            earliest_ms=DELIVERY_EARLIEST_MS,
        )
        security_id = f"{pair}@{contract_type}"
        rows = [
            {
                "security_id": security_id,
                "event_time": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
                "available_time": item[6],
                "contract_type": contract_type,
            }
            for item in payload
            if int(item[6]) <= now_ms
        ]
        if not rows:
            raise SourceError("Binance continuous returned only an in-progress kline")
        return normalize_ohlcv(rows, source=self.name, revision_id=f"{interval}.coinm")


class BinanceDeliveryUniverseSource(SourceAdapter):
    """COIN-M delivery-futures universe: contract listings + delivery dates.

    ``exchangeInfo`` only exposes contracts currently listed — a survivorship-
    truncated snapshot of deliverable quarterlies. Rows carry ``delivery_ms``
    (expiry) and ``onboard_ms`` (listing) so cash-and-carry lanes can align
    spot/future pairs on their true windows; ``available_time = now`` so PIT
    logic can never pretend the listing was known earlier.
    """

    name = "binance_delivery_universe"
    info_endpoint = "https://dapi.binance.com/dapi/v1/exchangeInfo"

    def fetch(
        self,
        *,
        base_asset: str | None = None,
        statuses: tuple[str, ...] = ("TRADING",),
    ) -> pl.DataFrame:
        info = self.client.get_json(self.info_endpoint)
        symbols = info.get("symbols") if isinstance(info, dict) else None
        if not isinstance(symbols, list):
            raise SourceError("Binance exchangeInfo response has no symbols list")
        wanted = frozenset(s.upper() for s in statuses)
        now = utc_now()
        rows: list[dict[str, Any]] = []
        for item in symbols:
            if not isinstance(item, dict):
                continue
            contract_type = str(item.get("contractType", "")).upper()
            if contract_type not in DELIVERY_CONTRACT_TYPES - {"PERPETUAL"}:
                continue
            if wanted and str(item.get("status", "")).upper() not in wanted:
                continue
            if (
                base_asset is not None
                and str(item.get("baseAsset", "")).upper() != base_asset.upper()
            ):
                continue
            symbol = str(item.get("symbol", "")).upper()
            if not symbol:
                continue
            try:
                delivery_ms = int(item["deliveryDate"])
                onboard_ms = int(item["onboardDate"])
            except KeyError as exc:
                raise SourceError(
                    f"delivery universe member {symbol!r} is missing a date field"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise SourceError(
                    f"delivery universe member {symbol!r} has invalid date fields"
                ) from exc
            if delivery_ms <= onboard_ms:
                raise SourceError(f"delivery universe member {symbol!r} delivers before it lists")
            rows.append(
                {
                    "security_id": symbol,
                    "event_time": delivery_ms,
                    "available_time": now,
                    "value": float(delivery_ms - onboard_ms) / 86_400_000.0,
                    "contract_type": contract_type,
                    "base_asset": str(item.get("baseAsset", "")).upper(),
                    "pair": str(item.get("pair", "")).upper(),
                    "onboard_ms": onboard_ms,
                    "delivery_ms": delivery_ms,
                }
            )
        rows.sort(key=lambda r: (r["delivery_ms"], r["security_id"]))
        if not rows:
            raise SourceError("Binance delivery universe resolved to zero contracts")
        return pit_frame(rows, source=self.name, revision_id="v1")


# Kraken futures contracts look like ``FI_XBTUSD_261225`` / ``PF_XBTUSD``:
# 2-letter family code (FI inverse dated, FF flexible dated, PI/PF perpetuals),
# an uppercase pair, and an optional dated tail.
_KRAKEN_CONTRACT_RE = re.compile(r"^[A-Z]{2}_[A-Z0-9]+(_[A-Z0-9]+)?$")

# Kraken public OHLC accepts these ``interval`` values (minutes).
_KRAKEN_SPOT_INTERVALS_MIN = frozenset({1, 5, 15, 30, 60, 240, 1440, 10080, 21600})

# Kraken charts resolutions -> candle span in ms, for in-progress filtering.
_KRAKEN_CHART_RESOLUTION_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


def _require_kraken_contract(symbol: str) -> str:
    """Kraken contract symbols look like ``FI_XBTUSD_261225`` / ``PF_XBTUSD``."""
    token = symbol.upper()
    if not _KRAKEN_CONTRACT_RE.match(token):
        raise ValueError(f"Kraken contract symbols look like 'FI_XBTUSD_261225', got {symbol!r}")
    return token


class KrakenSpotOhlcSource(SourceAdapter):
    """Kraken spot OHLC candles for one public pair (no API key needed).

    ``pair`` is the public pair code (``XBTUSD``, ``ETHUSD``); the response
    keys rows under an internal pair name (``XXBTZUSD``) we never assume —
    exactly one series per request is required or the fetch fails closed.
    Kraken appends the still-forming candle as the last row — its close
    (``time + interval``) lies in the future, so it is dropped the same way
    Binance's in-progress kline is.
    """

    name = "kraken_spot"
    endpoint = "https://api.kraken.com/0/public/OHLC"

    def fetch(
        self,
        *,
        pair: str = "XBTUSD",
        interval: int = 1440,
        since: int | None = None,
    ) -> pl.DataFrame:
        if not pair or not pair.upper().isalnum():
            raise ValueError(f"malformed Kraken pair {pair!r}")
        if interval not in _KRAKEN_SPOT_INTERVALS_MIN:
            raise ValueError(f"interval must be one of {sorted(_KRAKEN_SPOT_INTERVALS_MIN)}")
        pair = pair.upper()
        payload = self.client.get_json(
            query_url(self.endpoint, {"pair": pair, "interval": interval, "since": since})
        )
        if not isinstance(payload, dict):
            raise SourceError("Kraken OHLC response is not an object")
        errors = payload.get("error")
        if not isinstance(errors, list) or errors:
            raise SourceError(f"Kraken OHLC API error: {errors!r}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise SourceError("Kraken OHLC response has no result object")
        data_keys = [key for key in result if key != "last"]
        if len(data_keys) != 1:
            raise SourceError(f"Kraken OHLC returned {len(data_keys)} pair series, expected 1")
        candles = result[data_keys[0]]
        if not isinstance(candles, list) or not candles:
            raise SourceError(f"Kraken OHLC returned no candles for {pair}")
        now_s = int(utc_now().timestamp())
        span_s = interval * 60
        rows: list[dict[str, Any]] = []
        for item in candles:
            if not (isinstance(item, list) and len(item) >= 8):
                raise SourceError("malformed Kraken OHLC row")
            try:
                open_s = int(item[0])
            except (TypeError, ValueError) as exc:
                raise SourceError("malformed Kraken OHLC row") from exc
            if open_s + span_s > now_s:
                continue  # still-forming candle — its close time is in the future
            rows.append(
                {
                    "security_id": pair,
                    "event_time": open_s,
                    "open": item[1],
                    "high": item[2],
                    "low": item[3],
                    "close": item[4],
                    "volume": item[6],
                    "available_time": open_s + span_s,
                }
            )
        if not rows:
            raise SourceError(f"Kraken OHLC returned only an in-progress candle for {pair}")
        return normalize_ohlcv(rows, source=self.name, revision_id=f"{interval}m")


class KrakenFuturesMarkSource(SourceAdapter):
    """Kraken futures mark-price candles for one contract.

    The ``charts/v1`` ``mark`` series is a mark (index-derived fair price),
    not the last trade: for illiquid dated contracts the ``trade`` series is
    empty while mark always serves history, but mark != executed tape — the
    series is a settlement proxy, never fills. Kraken marks carry ``volume``
    ``'0'`` by construction.

    The still-forming candle (close time in the future) is dropped; an empty
    candle list fails closed — callers pick symbols from the
    ``kraken_futures_universe`` snapshot by ``lastTradingTime`` so expired or
    unlisted contracts never reach this fetch.
    """

    name = "kraken_futures_mark"
    endpoint = "https://futures.kraken.com/api/charts/v1/{tick_type}/{symbol}/{resolution}"

    def fetch(
        self,
        *,
        symbol: str,
        tick_type: str = "mark",
        resolution: str = "1d",
    ) -> pl.DataFrame:
        contract = _require_kraken_contract(symbol)
        if tick_type not in {"mark", "trade", "spot"}:
            raise ValueError("tick_type must be mark, trade, or spot")
        if resolution not in _KRAKEN_CHART_RESOLUTION_MS:
            raise ValueError(f"resolution must be one of {sorted(_KRAKEN_CHART_RESOLUTION_MS)}")
        url = self.endpoint.format(tick_type=tick_type, symbol=contract, resolution=resolution)
        payload = self.client.get_json(url)
        candles = payload.get("candles") if isinstance(payload, dict) else None
        if not isinstance(candles, list) or not candles:
            raise SourceError(f"Kraken charts returned no candles for {contract}")
        span_ms = _KRAKEN_CHART_RESOLUTION_MS[resolution]
        now_ms = int(utc_now().timestamp() * 1000)
        rows: list[dict[str, Any]] = []
        for item in candles:
            if not isinstance(item, dict):
                raise SourceError("malformed Kraken chart candle")
            try:
                open_ms = int(item["time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SourceError("malformed Kraken chart candle") from exc
            if open_ms + span_ms > now_ms:
                continue
            rows.append(
                {
                    "security_id": contract,
                    "event_time": open_ms,
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "close": item.get("close"),
                    "volume": item.get("volume", "0"),
                    "available_time": open_ms + span_ms,
                }
            )
        if not rows:
            raise SourceError(f"Kraken charts returned only an in-progress candle for {contract}")
        return normalize_ohlcv(rows, source=self.name, revision_id=f"{tick_type}.{resolution}")


class KrakenFuturesUniverseSource(SourceAdapter):
    """Kraken futures instrument universe: listings + delivery instants.

    ``instruments`` only exposes contracts currently listed — a survivorship-
    truncated snapshot, so expired dated contracts never appear and delivery
    instants for them must be derived elsewhere (e.g. the symbol tail). Row
    ``event_time`` is the snapshot instant (the listed state observed at
    ``serverTime``); the settlement instant rides as ``last_trading_time`` /
    ``last_trading_ms`` — null for perpetuals and other undated instruments.
    With ``include_undated`` off (default) undated instruments are filtered
    out — they have no delivery to anchor. ``available_time`` is the snapshot
    instant, so PIT logic can never pretend a listing was known earlier.
    """

    name = "kraken_futures_universe"
    endpoint = "https://futures.kraken.com/derivatives/api/v3/instruments"

    def fetch(self, *, include_undated: bool = False) -> pl.DataFrame:
        payload = self.client.get_json(self.endpoint)
        instruments = payload.get("instruments") if isinstance(payload, dict) else None
        if not isinstance(instruments, list):
            raise SourceError("Kraken instruments response has no instruments list")
        server_raw = payload.get("serverTime")
        now = utc_now()
        try:
            server_time = parse_time(server_raw) if server_raw is not None else now
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise SourceError("Kraken instruments serverTime is unparseable") from exc
        snapshot = min(server_time, now)  # clamp skew — future instants fail PIT
        rows: list[dict[str, Any]] = []
        for item in instruments:
            if not isinstance(item, dict):
                raise SourceError("malformed Kraken instrument entry")
            try:
                symbol = str(item["symbol"]).upper()
                instrument_type = str(item["type"])
                tradeable = bool(item["tradeable"])
            except KeyError as exc:
                raise SourceError("Kraken instrument entry is missing required fields") from exc
            # lastTradingTime is absent on perpetuals/undated instruments.
            last_trading = item.get("lastTradingTime")
            if not symbol or (last_trading is not None and not isinstance(last_trading, str)):
                raise SourceError(f"malformed Kraken instrument {symbol!r}")
            dated = last_trading is not None
            if dated:
                try:
                    delivery = parse_time(last_trading)
                except (TypeError, ValueError, OverflowError, OSError) as exc:
                    raise SourceError(
                        f"Kraken instrument {symbol!r} has unparseable lastTradingTime"
                    ) from exc
                delivery_ms = int(delivery.timestamp() * 1000)
                value = (delivery - snapshot).total_seconds() / 86_400.0
            else:
                if not include_undated:
                    continue
                delivery_ms = None
                value = None
            underlying = item.get("underlying")
            rows.append(
                {
                    "security_id": symbol,
                    "event_time": snapshot,
                    "available_time": snapshot,
                    "value": value,
                    "contract_type": instrument_type,
                    "underlying": str(underlying).upper() if underlying else None,
                    "last_trading_ms": delivery_ms,
                    "tradeable": tradeable,
                }
            )
        rows.sort(
            key=lambda r: (
                r["last_trading_ms"] if r["last_trading_ms"] is not None else 2**63,
                r["security_id"],
            )
        )
        if not rows:
            raise SourceError("Kraken futures universe resolved to zero instruments")
        return pit_frame(rows, source=self.name, revision_id="v1")


class KrakenFuturesFundingSource(SourceAdapter):
    """Kraken futures hourly funding-rate history for one contract.

    ``historical-funding-rates`` serves ``rates`` ascending by timestamp;
    each row carries ``fundingRate`` (absolute USD per contract) and
    ``relativeFundingRate`` (the per-period relative rate — the quantity
    comparable to other venues' realized rates), so ``value`` is the
    relative rate and the absolute one rides inside ``raw_value``. Rows
    become known only at their timestamp, so event_time == available_time.
    """

    name = "kraken_funding"
    endpoint = "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates"

    def fetch(self, *, symbol: str) -> pl.DataFrame:
        contract = _require_kraken_contract(symbol)
        payload = self.client.get_json(query_url(self.endpoint, {"symbol": contract}))
        if not isinstance(payload, dict):
            raise SourceError("Kraken funding response is not an object")
        if payload.get("result") != "success":
            raise SourceError(
                f"Kraken funding API error: {payload.get('errors') or payload.get('status')!r}"
            )
        rates = payload.get("rates")
        if not isinstance(rates, list) or not rates:
            raise SourceError(f"Kraken funding returned no rates for {contract}")
        rows: list[dict[str, Any]] = []
        last_ms = -1
        for item in rates:
            if not isinstance(item, dict):
                raise SourceError("malformed Kraken funding row")
            try:
                stamp = parse_time(item["timestamp"])
                rate = float(item["relativeFundingRate"])
            except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
                raise SourceError("malformed Kraken funding row") from exc
            stamp_ms = int(stamp.timestamp() * 1000)
            if stamp_ms <= last_ms:
                raise SourceError("Kraken funding timestamps are not strictly increasing")
            last_ms = stamp_ms
            if not math.isfinite(rate) or abs(rate) >= 10.0:
                raise SourceError("Kraken funding rate is not finite")
            rows.append(
                {
                    "security_id": contract,
                    "event_time": stamp_ms,
                    "available_time": stamp_ms,
                    "value": rate,
                    "raw_value": json.dumps(item, sort_keys=True),
                }
            )
        return normalize_observations(rows, source=self.name, revision_id="v1")


# OKX public REST wraps every payload as ``{"code": "0", "msg": "", "data":
# [...]}`` — the adapters below unwrap ``data`` and fail closed on a nonzero
# ``code`` or any schema drift. Instrument ids look like ``BTC-USDT`` (spot),
# ``BTC-USDT-SWAP`` (linear perp), ``BTC-USD-260925`` (dated future).
_OKX_INST_RE = re.compile(r"^[A-Z0-9]+(-[A-Z0-9]+){1,3}$")

# OKX ``bar`` values -> candle span in ms. ``*utc`` variants anchor to UTC
# boundaries; spans feed ``available_time`` only — in-progress candles are
# identified by OKX's own trailing ``confirm`` flag, not the clock.
_OKX_BAR_SPAN_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1H": 3_600_000,
    "2H": 7_200_000,
    "4H": 14_400_000,
    "6H": 21_600_000,
    "12H": 43_200_000,
    "1D": 86_400_000,
    "1Dutc": 86_400_000,
    "2Dutc": 172_800_000,
    "1W": 604_800_000,
    "1Wutc": 604_800_000,
}

#: Per-endpoint ``limit`` ceilings from the OKX public docs.
_OKX_CANDLE_LIMIT = 300
_OKX_FUNDING_LIMIT = 100


def _require_okx_inst(inst_id: str) -> str:
    """OKX instrument ids look like ``BTC-USDT`` / ``BTC-USDT-SWAP``."""
    token = inst_id.upper()
    if not _OKX_INST_RE.match(token):
        raise ValueError(
            f"OKX instrument ids look like 'BTC-USDT' or 'BTC-USDT-SWAP', got {inst_id!r}"
        )
    return token


def _require_okx_bar(bar: str) -> str:
    if bar not in _OKX_BAR_SPAN_MS:
        raise ValueError(f"bar must be one of {sorted(_OKX_BAR_SPAN_MS)}")
    return bar


def _okx_optional_ms(value: Any) -> int | None:
    """OKX ms-epoch string field that may be empty; fails closed on garbage."""
    text = str(value or "")
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError) as exc:
        raise SourceError("OKX instrument has an unparseable ms-epoch field") from exc


def _okx_data(payload: Any, *, what: str) -> Any:
    """Unwrap OKX's ``{code, msg, data}`` envelope, failing closed."""
    if not isinstance(payload, dict):
        raise SourceError(f"OKX {what} response is not an object")
    code = payload.get("code")
    if code != "0":
        raise SourceError(f"OKX {what} API error: code={code!r} msg={payload.get('msg')!r}")
    if "data" not in payload:
        raise SourceError(f"OKX {what} response has no data field")
    return payload["data"]


def _okx_candle_frame(
    payload: Any,
    *,
    inst_id: str,
    fields: int,
    what: str,
) -> list[list[Any]]:
    """Validate one page of OKX candle rows (``[ts, o, h, l, c, ..., confirm]``)."""
    data = _okx_data(payload, what=what)
    if not isinstance(data, list):
        raise SourceError(f"OKX {what} data is not a list for {inst_id}")
    for item in data:
        if not (isinstance(item, list) and len(item) >= fields):
            raise SourceError(f"malformed OKX {what} candle row")
        try:
            int(item[0])
        except (TypeError, ValueError) as exc:
            raise SourceError(f"malformed OKX {what} candle row") from exc
    return data


class OkxSpotOhlcSource(SourceAdapter):
    """OKX spot OHLC candles for one instrument (no API key needed).

    Rows arrive newest-first as ``[ts, o, h, l, c, vol, volCcy,
    volCcyQuote, confirm]``; the still-forming candle carries
    ``confirm != "1"`` and is dropped — never rewritten — the same way
    Kraken's/Binance's in-progress rows are. Pagination walks backward via
    ``after=<oldest open ts>`` (OKX returns the newest page first).
    """

    name = "okx_spot"
    endpoint = "https://www.okx.com/api/v5/market/candles"

    def fetch(
        self,
        *,
        inst_id: str = "BTC-USDT",
        bar: str = "1Dutc",
        limit: int = 100,
        max_pages: int = 1,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        inst = _require_okx_inst(inst_id)
        _require_okx_bar(bar)
        if not 1 <= limit <= _OKX_CANDLE_LIMIT:
            raise ValueError(f"limit must be between 1 and {_OKX_CANDLE_LIMIT}")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        span_ms = _OKX_BAR_SPAN_MS[bar]
        rows: list[dict[str, Any]] = []
        seen_ts: set[int] = set()
        cursor: int | None = None
        for page in range(max_pages):
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {"instId": inst, "bar": bar, "limit": limit, "after": cursor},
                )
            )
            data = _okx_candle_frame(payload, inst_id=inst, fields=9, what="candles")
            if not data:
                if page == 0:
                    raise SourceError(f"OKX candles returned no rows for {inst}")
                break
            oldest = min(int(item[0]) for item in data)
            for item in data:
                open_ms = int(item[0])
                if open_ms in seen_ts:
                    continue  # overlap row on a page boundary
                seen_ts.add(open_ms)
                if str(item[8]) != "1":
                    continue  # still-forming candle per OKX's own confirm flag
                rows.append(
                    {
                        "security_id": inst,
                        "event_time": open_ms,
                        "open": item[1],
                        "high": item[2],
                        "low": item[3],
                        "close": item[4],
                        "volume": item[5],
                        "available_time": open_ms + span_ms,
                    }
                )
            if len(data) < limit:
                break
            cursor = oldest
            if pause_seconds > 0:
                time.sleep(pause_seconds)
        if not rows:
            raise SourceError(f"OKX candles returned only in-progress rows for {inst}")
        return normalize_ohlcv(rows, source=self.name, revision_id=bar)


class OkxMarkCandlesSource(SourceAdapter):
    """OKX mark-price candles for one derivative instrument.

    The ``mark-price-candles`` series is an index-derived fair price, not
    last trade — same settlement-proxy caveat as ``kraken_futures_mark``.
    Rows are ``[ts, o, h, l, c, confirm]``: no volume exists for a mark
    series, so ``volume`` is reported as ``"0"`` by construction.
    """

    name = "okx_mark"
    endpoint = "https://www.okx.com/api/v5/market/mark-price-candles"

    def fetch(
        self,
        *,
        inst_id: str,
        bar: str = "1Dutc",
        limit: int = 100,
        max_pages: int = 1,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        inst = _require_okx_inst(inst_id)
        _require_okx_bar(bar)
        if not 1 <= limit <= _OKX_CANDLE_LIMIT:
            raise ValueError(f"limit must be between 1 and {_OKX_CANDLE_LIMIT}")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        span_ms = _OKX_BAR_SPAN_MS[bar]
        rows: list[dict[str, Any]] = []
        seen_ts: set[int] = set()
        cursor: int | None = None
        for page in range(max_pages):
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {"instId": inst, "bar": bar, "limit": limit, "after": cursor},
                )
            )
            data = _okx_candle_frame(payload, inst_id=inst, fields=6, what="mark-price-candles")
            if not data:
                if page == 0:
                    raise SourceError(f"OKX mark candles returned no rows for {inst}")
                break
            oldest = min(int(item[0]) for item in data)
            for item in data:
                open_ms = int(item[0])
                if open_ms in seen_ts:
                    continue
                seen_ts.add(open_ms)
                if str(item[5]) != "1":
                    continue
                rows.append(
                    {
                        "security_id": inst,
                        "event_time": open_ms,
                        "open": item[1],
                        "high": item[2],
                        "low": item[3],
                        "close": item[4],
                        "volume": "0",
                        "available_time": open_ms + span_ms,
                    }
                )
            if len(data) < limit:
                break
            cursor = oldest
            if pause_seconds > 0:
                time.sleep(pause_seconds)
        if not rows:
            raise SourceError(f"OKX mark candles returned only in-progress rows for {inst}")
        return normalize_ohlcv(rows, source=self.name, revision_id=f"mark.{bar}")


class OkxFundingHistorySource(SourceAdapter):
    """OKX realized funding-rate history for one perpetual swap.

    ``funding-rate-history`` serves the *realized* per-settlement rate
    (``realizedRate``; ``fundingRate`` is the same charge on settled rows),
    newest-first, settling at ``fundingTime`` — typically every 8h. A rate
    is only fixed at ``fundingTime``, so event_time == available_time and
    rows stamped in the future are dropped. Pagination walks backward via
    ``after=<oldest fundingTime>``.
    """

    name = "okx_funding"
    endpoint = "https://www.okx.com/api/v5/public/funding-rate-history"

    def fetch(
        self,
        *,
        inst_id: str = "BTC-USDT-SWAP",
        limit: int = 100,
        max_pages: int = 10,
        pause_seconds: float = 0.3,
    ) -> pl.DataFrame:
        inst = _require_okx_inst(inst_id)
        if not 1 <= limit <= _OKX_FUNDING_LIMIT:
            raise ValueError(f"limit must be between 1 and {_OKX_FUNDING_LIMIT}")
        if max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        now_ms = int(utc_now().timestamp() * 1000)
        rows: list[dict[str, Any]] = []
        seen_ts: set[int] = set()
        cursor: int | None = None
        for page in range(max_pages):
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {"instId": inst, "limit": limit, "after": cursor},
                )
            )
            data = _okx_data(payload, what="funding-rate-history")
            if not isinstance(data, list):
                raise SourceError(f"OKX funding data is not a list for {inst}")
            if not data:
                if page == 0:
                    raise SourceError(f"OKX funding returned no rows for {inst}")
                break
            page_times: list[int] = []
            for item in data:
                if not isinstance(item, dict):
                    raise SourceError("malformed OKX funding row")
                try:
                    funding_time = int(item["fundingTime"])
                    rate_raw = item.get("realizedRate", item.get("fundingRate"))
                    if rate_raw is None:
                        raise KeyError("realizedRate/fundingRate")
                    rate = float(rate_raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise SourceError("malformed OKX funding row") from exc
                page_times.append(funding_time)
                if funding_time in seen_ts:
                    continue
                seen_ts.add(funding_time)
                if funding_time > now_ms:
                    continue  # not yet charged — the rate fixes at fundingTime
                if not math.isfinite(rate) or abs(rate) >= 10.0:
                    raise SourceError("OKX funding rate is not finite")
                rows.append(
                    {
                        "security_id": str(item.get("instId", inst)).upper(),
                        "event_time": funding_time,
                        "available_time": funding_time,
                        "value": rate,
                        "raw_value": json.dumps(item, sort_keys=True),
                    }
                )
            if len(data) < limit:
                break
            cursor = min(page_times)
            if pause_seconds > 0:
                time.sleep(pause_seconds)
        if not rows:
            raise SourceError(f"OKX funding returned no realized rows for {inst}")
        return normalize_observations(rows, source=self.name, revision_id="v1")


class OkxSwapUniverseSource(SourceAdapter):
    """OKX perpetual-swap universe: listing instants + contract metadata.

    ``public/instruments?instType=SWAP`` is a survivorship-truncated snapshot
    — it lists only currently served instruments, so a delisted perp never
    appears. ``event_time``/``available_time`` are the collection instant
    (the endpoint returns no server clock); ``value`` is null — contract
    metadata rides in the metadata columns. ``live_only`` (default) keeps
    ``state == "live"`` rows.
    """

    name = "okx_swap_universe"
    endpoint = "https://www.okx.com/api/v5/public/instruments"

    def fetch(self, *, live_only: bool = True) -> pl.DataFrame:
        payload = self.client.get_json(query_url(self.endpoint, {"instType": "SWAP"}))
        data = _okx_data(payload, what="instruments")
        if not isinstance(data, list) or not data:
            raise SourceError("OKX instruments returned no instruments")
        snapshot = utc_now()
        rows: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                raise SourceError("malformed OKX instrument entry")
            try:
                inst_id = str(item["instId"]).upper()
                state = str(item["state"])
                family = str(item["instFamily"])
            except KeyError as exc:
                raise SourceError("OKX instrument entry is missing required fields") from exc
            if not inst_id or not _OKX_INST_RE.match(inst_id):
                raise SourceError(f"malformed OKX instrument id {inst_id!r}")
            if live_only and state != "live":
                continue
            rows.append(
                {
                    "security_id": inst_id,
                    "event_time": snapshot,
                    "available_time": snapshot,
                    "value": None,
                    "inst_family": family,
                    "ct_type": item.get("ctType"),
                    "ct_val": item.get("ctVal"),
                    "ct_val_ccy": item.get("ctValCcy"),
                    "settle_ccy": item.get("settleCcy"),
                    "tick_sz": item.get("tickSz"),
                    "lot_sz": item.get("lotSz"),
                    "list_time_ms": _okx_optional_ms(item.get("listTime")),
                    "state": state,
                }
            )
        rows.sort(key=lambda r: r["security_id"])
        if not rows:
            raise SourceError("OKX swap universe resolved to zero instruments")
        return pit_frame(rows, source=self.name, revision_id="v1")


class GdeltSource(SourceAdapter):
    name = "gdelt"
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"

    def fetch(
        self, *, query: str, mode: str = "timelinevol", format: str = "json", maxrecords: int = 250
    ) -> pl.DataFrame:
        payload = self.client.get_json(
            query_url(
                self.endpoint,
                {"query": query, "mode": mode, "format": format, "maxrecords": maxrecords},
            )
        )
        rows = payload.get("timeline") if isinstance(payload, dict) else None
        if not rows:
            return pl.DataFrame()
        points = rows[0].get("data", rows) if isinstance(rows[0], dict) else rows
        if not isinstance(points, list):
            raise SourceError("GDELT timeline payload is not a list")
        flattened = []
        for point in points:
            flattened.append(
                {
                    "security_id": query,
                    "event_time": point.get("date"),
                    "available_time": utc_now(),
                    "value": point.get("value"),
                    "raw_value": json.dumps(point, sort_keys=True),
                }
            )
        return normalize_observations(flattened, source=self.name, value_key="value")


class SecEdgarSource(SourceAdapter):
    name = "sec_edgar"
    endpoint = "https://data.sec.gov/submissions/CIK{cik}.json"

    def fetch(self, *, cik: str) -> pl.DataFrame:
        normalized = str(cik).strip().upper().removeprefix("CIK")
        if not normalized.isdigit():
            raise ValueError("SEC CIK must contain only digits")
        payload = self.client.get_json(
            self.endpoint.format(cik=normalized.zfill(10)),
            headers={"Accept-Encoding": "gzip, deflate"},
        )
        recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, dict) else {}
        filing_dates = recent.get("filingDate", [])
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        if not (len(filing_dates) == len(forms) == len(accessions)):
            raise SourceError("SEC submissions payload has ragged filing columns")
        rows = [
            {
                "security_id": normalized,
                "event_time": filed,
                "available_time": utc_now(),
                "value": form,
                "title": accession,
            }
            for filed, form, accession in zip(
                filing_dates,
                forms,
                accessions,
                strict=True,
            )
        ]
        return (
            normalize_observations(rows, source=self.name, value_key="value")
            if rows
            else pl.DataFrame()
        )


class FredSource(SourceAdapter):
    name = "fred"
    endpoint = "https://api.stlouisfed.org/fred/series/observations"
    csv_endpoint = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def fetch(
        self,
        *,
        series_id: str,
        api_key: str | None = None,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pl.DataFrame:
        if api_key or os.getenv("FRED_API_KEY"):
            payload = self.client.get_json(
                query_url(
                    self.endpoint,
                    {
                        "series_id": series_id,
                        "api_key": api_key or os.getenv("FRED_API_KEY"),
                        "file_type": "json",
                        "observation_start": observation_start,
                        "observation_end": observation_end,
                    },
                )
            )
            rows = [
                {
                    "security_id": series_id,
                    "event_time": item.get("date"),
                    "available_time": utc_now(),
                    "value": item.get("value"),
                }
                for item in payload.get("observations", [])
            ]
        else:
            text = self.client.get_text(
                query_url(
                    self.csv_endpoint,
                    {"id": series_id, "cosd": observation_start, "coed": observation_end},
                )
            )
            table = csv_rows(text)
            # ALFRED graph CSV data columns are vintage-suffixed
            # ("GDP_20260915"); FRED uses the bare series id. A single-series
            # CSV carries exactly one data column besides observation_date.
            value_col = series_id
            if table and series_id not in table[0]:
                candidates = [key for key in table[0] if key != "observation_date"]
                if len(candidates) != 1:
                    raise SourceError(f"CSV response has no unambiguous {series_id!r} data column")
                value_col = candidates[0]
            rows = [
                {
                    "security_id": series_id,
                    "event_time": row.get("observation_date"),
                    "available_time": utc_now(),
                    "value": row.get(value_col),
                }
                for row in table
            ]
        return normalize_observations(rows, source=self.name)


class AlfredSource(FredSource):
    name = "alfred"
    csv_endpoint = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


class TreasurySource(SourceAdapter):
    name = "us_treasury"
    endpoint = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

    def fetch(
        self,
        *,
        endpoint: str = "v2/accounting/od/avg_interest_rates",
        filters: str | None = None,
        page_size: int = 1000,
    ) -> pl.DataFrame:
        payload = self.client.get_json(
            query_url(
                f"{self.endpoint}/{endpoint}",
                {"filter": filters, "page[size]": page_size, "format": "json"},
            )
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not rows:
            return pl.DataFrame()
        date_key = next(
            (key for key in ("record_date", "effective_date", "date") if key in rows[0]), None
        )
        value_key = next(
            (key for key in ("avg_interest_rate_amt", "value", "total") if key in rows[0]), None
        )
        if not date_key or not value_key:
            raise SourceError("Treasury payload has no recognized date/value fields")
        return normalize_observations(
            [
                {
                    "security_id": str(row.get("security_type", endpoint)),
                    # Ragged later rows surface as a null timestamp, which
                    # normalize_observations rejects — never silently skipped.
                    "event_time": row.get(date_key),
                    "available_time": utc_now(),
                    "value": row.get(value_key),
                }
                for row in rows
            ],
            source=self.name,
        )


class CftcSource(SourceAdapter):
    name = "cftc_cot"
    endpoint = "https://publicreporting.cftc.gov/resource/kh3c-gbw2.json"

    def fetch(self, *, limit: int = 5000, query: str | None = None) -> pl.DataFrame:
        payload = self.client.get_json(query_url(self.endpoint, {"$limit": limit, "$query": query}))
        if not isinstance(payload, list):
            raise SourceError("CFTC payload must be a list")
        date_key = next(
            (
                key
                for key in ("report_date_as_yyyymmdd", "report_date")
                if payload and key in payload[0]
            ),
            None,
        )
        value_key = next(
            (
                key
                for key in ("open_interest_all", "noncomm_positions_long_all")
                if payload and key in payload[0]
            ),
            None,
        )
        if not date_key or not value_key:
            return pl.DataFrame()
        return normalize_observations(
            [
                {
                    "security_id": row.get("market_and_exchange_names", self.name),
                    "event_time": row.get(date_key),
                    "available_time": utc_now(),
                    "value": row.get(value_key),
                }
                for row in payload
            ],
            source=self.name,
        )


class FinaSource(SourceAdapter):
    name = "finra_short_sale_volume"
    endpoint = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"

    def fetch(self, *, url: str | None = None, limit: int = 1000) -> pl.DataFrame:
        payload = self.client.get_json(query_url(url or self.endpoint, {"limit": limit}))
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise SourceError("FINRA payload must be a list")
        date_key = next((key for key in ("tradeDate", "date") if rows and key in rows[0]), None)
        value_key = next(
            (key for key in ("shortVolume", "shortVolumeExempt") if rows and key in rows[0]), None
        )
        if not date_key or not value_key:
            return pl.DataFrame()
        return normalize_observations(
            [
                {
                    "security_id": row.get("symbol", self.name),
                    "event_time": row.get(date_key),
                    "available_time": utc_now(),
                    "value": row.get(value_key),
                }
                for row in rows
            ],
            source=self.name,
        )


class WorldBankSource(SourceAdapter):
    name = "world_bank"
    endpoint = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

    def fetch(
        self, *, country: str = "US", indicator: str = "NY.GDP.MKTP.CD", page_size: int = 1000
    ) -> pl.DataFrame:
        payload = self.client.get_json(
            query_url(
                self.endpoint.format(country=country, indicator=indicator),
                {"format": "json", "per_page": page_size},
            )
        )
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            raise SourceError("World Bank payload has unexpected shape")
        return normalize_observations(
            [
                {
                    "security_id": f"{country}:{indicator}",
                    "event_time": f"{row.get('date')}-12-31",
                    "available_time": utc_now(),
                    "value": row.get("value"),
                }
                for row in payload[1]
                if row.get("value") is not None
            ],
            source=self.name,
        )


_BEA_QUARTER_END = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}


def _bea_period_date(period: str) -> str:
    """Map a BEA TimePeriod (annual/quarterly/monthly) to a period-end date."""
    if "Q" in period:
        year, _, quarter = period.partition("Q")
        end = _BEA_QUARTER_END.get(f"Q{quarter}")
        if end:
            return f"{year}-{end}"
    elif "M" in period:
        year, _, month = period.partition("M")
        if month.isdigit() and 1 <= int(month) <= 12:
            _, last = calendar.monthrange(int(year), int(month))
            return f"{year}-{month}-{last:02d}"
    if period.isdigit() and len(period) == 4:
        return f"{period}-12-31"
    return period


class BeaSource(SourceAdapter):
    name = "bea"
    endpoint = "https://apps.bea.gov/api/data/"

    def fetch(
        self,
        *,
        dataset: str = "NIPA",
        table_name: str = "T10101",
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> pl.DataFrame:
        key = api_key or os.getenv("BEA_API_KEY") or user_id
        if not key:
            raise SourceError("BEA requires api_key or BEA_API_KEY; credentials are never embedded")
        payload = self.client.get_json(
            query_url(
                self.endpoint,
                {
                    "UserID": key,
                    "method": "GETDATA",
                    "datasetname": dataset,
                    "TableName": table_name,
                    "ResultFormat": "JSON",
                },
            )
        )
        data = (
            payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
            if isinstance(payload, dict)
            else []
        )
        rows = [
            {
                "security_id": table_name,
                "event_time": _bea_period_date(row.get("TimePeriod", "")),
                "available_time": utc_now(),
                "value": row.get("DataValue"),
            }
            for row in data
            if row.get("TimePeriod") and row.get("DataValue") not in (None, "")
        ]
        return normalize_observations(rows, source=self.name) if rows else pl.DataFrame()


class ItchSampleSource(SourceAdapter):
    name = "nasdaq_itch"

    def fetch(self, *, path: str | Path) -> pl.DataFrame:
        text = Path(path).read_text(encoding="utf-8")
        rows = csv_rows(text)
        return normalize_ohlcv(rows, source=self.name, symbol_key="security_id")


class Fi2010Source(SourceAdapter):
    name = "fi_2010"

    def fetch(self, *, path: str | Path) -> pl.DataFrame:
        frame = pl.read_csv(Path(path), separator=",", try_parse_dates=True)
        if frame.is_empty():
            return frame
        if "event_time" not in frame.columns:
            raise SourceError("FI-2010 file must include event_time")
        return frame.with_columns(pl.lit(self.name).alias("source"))


class _OptionalLibrarySource(SourceAdapter):
    library_name = "optional"

    def fetch(self, *, payload: Any = None, **kwargs: Any) -> pl.DataFrame:
        if payload is not None:
            if isinstance(payload, pl.DataFrame):
                return payload
            if isinstance(payload, list):
                return pl.DataFrame(payload)
        raise SourceError(
            f"{self.name} requires an explicit {self.library_name} payload; no dependency is installed by default"
        )


def _vendor_names(
    names: object, default: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """Coerce ``names`` to ``(security_id, vendor_symbol)`` pairs; ``None`` picks ``default``."""
    if names is None:
        return default
    if isinstance(names, str):
        pairs: list[tuple[str, str]] = []
        for item in names.split(","):
            security_id, sep, symbol = item.partition(":")
            if not sep or not security_id.strip() or not symbol.strip():
                raise ValueError(f"names entries must be 'SECURITY:vendor_symbol', got {item!r}")
            pairs.append((security_id.strip(), symbol.strip()))
        return tuple(pairs)
    if isinstance(names, (list, tuple)):
        pairs = []
        for entry in names:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError(
                    f"names entries must be (security_id, vendor_symbol) pairs, got {entry!r}"
                )
            security_id, symbol = entry
            pairs.append((str(security_id), str(symbol)))
        return tuple(pairs)
    raise ValueError(f"names must be a comma string or pair sequence, got {type(names).__name__}")


def _vendor_bool(value: object) -> bool:
    """Coerce CLI ``--param`` strings to bool; fail closed on anything else."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"strict must be a boolean, got {value!r}")


class StooqSource(SourceAdapter):
    """Stooq daily-equities universe as a PIT-stamped source frame.

    ``fetch`` returns a session-close PIT frame for a comma list of
    ``SECURITY:stooq_symbol`` names (default US_LIQUID). Per-name fetch
    failures fail the whole call under ``strict=True`` (the default) so a
    silently-truncated universe can never reach the pipeline; pass
    ``strict=false`` to keep the surviving names. ``start``/``end`` are
    session dates (ISO strings), not instants.
    """

    name = "stooq"

    def fetch(
        self,
        *,
        names: object = None,
        start: object = None,
        end: object = None,
        pause_s: float = 0.4,
        max_workers: int = 4,
        strict: object = True,
    ) -> pl.DataFrame:
        # Lazy: data.adapters.__init__ already imports sources.base, so a
        # top-level edge here would create a module cycle through the registry.
        from quant_fund.data.adapters.stooq import US_LIQUID, collect_bars

        universe = _vendor_names(names, US_LIQUID)
        strict_flag = _vendor_bool(strict)
        start_dt = None if start is None else parse_time(start)
        end_dt = None if end is None else parse_time(end)
        bars, errors = collect_bars(
            universe,
            start=start_dt,
            end=end_dt,
            pause_s=float(pause_s),
            max_workers=int(max_workers),
        )
        if errors and strict_flag:
            raise SourceError(
                f"stooq fetch failed for {len(errors)}/{len(universe)} names: {errors}"
            )
        if bars.is_empty():
            raise SourceError(
                f"stooq fetch returned no bars ({len(errors)}/{len(universe)} names failed)"
            )
        return bars


class YahooSource(SourceAdapter):
    """Yahoo v8 daily-equities universe as a PIT-stamped source frame.

    Same contract as :class:`StooqSource` but over Yahoo chart payloads
    (``SECURITY:yahoo_symbol`` names, e.g. ``AAPL:AAPL`` or ``VOD:VOD.L``;
    default YAHOO_US). Kept as a separate registered source on purpose — the
    ``source`` column must name the true vendor, so there is no silent
    stooq→yahoo fallback.
    """

    name = "yahoo"

    def fetch(
        self,
        *,
        names: object = None,
        start: object = None,
        end: object = None,
        pause_s: float = 0.15,
        max_workers: int = 4,
        strict: object = True,
    ) -> pl.DataFrame:
        # Lazy: same module-cycle reason as StooqSource.
        from quant_fund.data.adapters.yahoo_eod import YAHOO_US, collect_bars

        universe = _vendor_names(names, YAHOO_US)
        strict_flag = _vendor_bool(strict)
        start_dt = None if start is None else parse_time(start)
        end_dt = None if end is None else parse_time(end)
        bars, errors = collect_bars(
            universe,
            start=start_dt,
            end=end_dt,
            pause_s=float(pause_s),
            max_workers=int(max_workers),
        )
        if errors and strict_flag:
            raise SourceError(
                f"yahoo fetch failed for {len(errors)}/{len(universe)} names: {errors}"
            )
        if bars.is_empty():
            raise SourceError(
                f"yahoo fetch returned no bars ({len(errors)}/{len(universe)} names failed)"
            )
        return bars


class CryptofeedSource(_OptionalLibrarySource):
    name = "cryptofeed"
    library_name = "cryptofeed"


class CcxtSource(_OptionalLibrarySource):
    name = "ccxt"
    library_name = "ccxt"


class OpenBBSource(_OptionalLibrarySource):
    name = "openbb"
    library_name = "openbb"
