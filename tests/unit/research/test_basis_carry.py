"""basis_carry: settlement-anchored cash-and-carry bench (P5.3).

Covers the Kraken source adapters with canned fixtures (in-progress candle
dropped, malformed payloads fail closed), the bench's known-answer path on
a planted annualized carry, fail-closed input cases, the sealed
``basis_carry.v1`` receipt contract, and the CLI surface. All frames here
are SYNTHETIC fixtures — correctness evidence, never market evidence.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant_fund.data.sources.base import SourceError
from quant_fund.research.basis_carry import (
    CarryContractInput,
    basis_carry_contract_errors,
    kraken_delivery_from_symbol,
    run_basis_carry,
    write_basis_carry_receipt,
)

pytestmark = pytest.mark.synthetic


def _frame(dates: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"event_time": dates, "close": closes})


def _planted_carry(
    days: int = 120,
    *,
    spot_price: float = 100.0,
    annualized: float = 0.05,
    delivery: datetime,
    start: date = date(2025, 1, 1),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Spot flat; future = S * (1 + annualized * dte/365) — a known carry path."""
    dates = [start + timedelta(days=i) for i in range(days)]
    spot = [spot_price] * days
    future = []
    for day in dates:
        dte = (delivery - datetime.combine(day, time(0, 0), tzinfo=UTC)).total_seconds() / 86_400.0
        future.append(spot_price * (1.0 + annualized * dte / 365.0))
    return _frame(dates, spot), _frame(dates, future)


DELIVERY = datetime(2025, 9, 26, 16, 0, tzinfo=UTC)  # FI_XBTUSD_250926


def _run(days: int = 120, **kwargs: Any):
    spot, future = _planted_carry(days=days, delivery=DELIVERY)
    contracts = [
        CarryContractInput(
            symbol="FI_XBTUSD_250926",
            frame=future,
            delivery=DELIVERY,
            data_label="SYNTHETIC",
        )
    ]
    return run_basis_carry(spot=spot, spot_label="SYNTHETIC", contracts=contracts, **kwargs)


class TestKrakenDeliveryFromSymbol:
    def test_fi_and_ff_families(self) -> None:
        assert kraken_delivery_from_symbol("FI_XBTUSD_261225") == datetime(
            2026, 12, 25, 16, 0, tzinfo=UTC
        )
        assert kraken_delivery_from_symbol("FF_XBTUSD_261030") == datetime(
            2026, 10, 30, 8, 0, tzinfo=UTC
        )

    @pytest.mark.parametrize(
        "bad", ["XBTUSD", "FI_XBTUSD", "ZZ_XBTUSD_261225", "FI_XBTUSD_26134", ""]
    )
    def test_malformed_symbols_fail(self, bad: str) -> None:
        with pytest.raises(ValueError):
            kraken_delivery_from_symbol(bad)


class TestKnownAnswer:
    def test_constant_annualized_carry_recovers_exactly(self) -> None:
        _frame_out, receipt = _run()
        row = receipt["results"][0]
        assert row["status"] == "ok"
        # Planted: F = S(1 + 0.05 * dte/365) -> ann basis == 0.05 everywhere.
        assert math.isclose(row["ann_basis_mean"], 0.05, rel_tol=1e-9)
        assert math.isclose(row["ann_basis_std"], 0.0, abs_tol=1e-9)
        # Terminal: last shared date 2025-04-30 is 149d before delivery, so
        # convergence_residual == 0.05 * dte_last / 365 (not zero — contract
        # still listed in the fixture window; the field stays honest).
        expected_dte = (
            DELIVERY - datetime.combine(date(2025, 4, 30), time(0, 0), tzinfo=UTC)
        ).total_seconds() / 86_400.0
        assert math.isclose(row["residual_dte_days"], expected_dte, rel_tol=1e-9)
        assert math.isclose(row["convergence_residual"], 0.05 * expected_dte / 365.0)
        assert row["delivered"] is False
        assert row["basis_diff_std"] is not None and row["basis_diff_std"] >= 0.0
        assert row["n_dates"] == 120

    def test_bucket_means_use_nearest_observation_at_or_below(self) -> None:
        # Window runs through the delivery date: dte spans ~268 .. 16/24.
        _frame_out, receipt = _run(days=269)
        buckets = receipt["results"][0]["dte_buckets"]
        # bucket 90 -> nearest obs inside 90d: 2025-06-28, dte = 89 + 16/24.
        assert buckets["90"]["dte_days"] == pytest.approx(89.0 + 16.0 / 24.0)
        assert buckets["90"]["basis"] == pytest.approx(0.05 * (89.0 + 16.0 / 24.0) / 365.0)
        assert buckets["90"]["ann_basis"] == pytest.approx(0.05)
        # bucket 1 -> the delivery-date obs itself (dte 16/24 < 1): the raw
        # basis rides along, annualization is skipped under the dte floor.
        assert buckets["1"]["dte_days"] == pytest.approx(16.0 / 24.0)
        assert buckets["1"]["ann_basis"] is None
        assert buckets["30"]["date"] == "2025-08-28"  # dte 29.667 ≤ 30

    def test_delivered_contract_records_terminal_residual(self) -> None:
        # Contract delivered inside the window: last merged date is the
        # delivery date itself, residual is the true settlement anchor.
        spot, future = _planted_carry(days=87, delivery=DELIVERY, start=date(2025, 7, 2))
        frame, receipt = run_basis_carry(
            spot=spot,
            spot_label="SYNTHETIC",
            contracts=[
                CarryContractInput(
                    symbol="FI_XBTUSD_250926",
                    frame=future,
                    delivery=DELIVERY,
                    data_label="SYNTHETIC",
                )
            ],
        )
        row = receipt["results"][0]
        assert row["status"] == "ok"
        assert row["delivered"] is True
        assert row["last_date"] == "2025-09-26"
        # dte at the delivery-date obs is 16/24 (the 16:00Z settlement).
        assert row["residual_dte_days"] == pytest.approx(16.0 / 24.0)
        assert row["convergence_residual"] == pytest.approx(0.05 * (16.0 / 24.0) / 365.0)
        assert frame["delivered"].to_list() == [True]

    def test_dte_floor_drops_sub_day_rows_from_annualization(self) -> None:
        # Last obs is the delivery date itself (dte = 16/24 < 1): it counts in
        # the basis path + residual but not in the annualized aggregates.
        _frame_out, receipt = _run(days=87, **{})
        row = receipt["results"][0]
        # Fixture window ends 2025-03-28, far before delivery — every row is
        # inside the annualization floor.
        assert row["ann_basis_n"] == row["n_dates"]
        spot, future = _planted_carry(days=87, delivery=DELIVERY, start=date(2025, 7, 2))
        _, receipt2 = run_basis_carry(
            spot=spot,
            spot_label="SYNTHETIC",
            contracts=[
                CarryContractInput(
                    symbol="FI_XBTUSD_250926",
                    frame=future,
                    delivery=DELIVERY,
                    data_label="SYNTHETIC",
                )
            ],
        )
        row2 = receipt2["results"][0]
        assert row2["n_dates"] == 87
        assert row2["ann_basis_n"] == 86  # delivery-day obs (dte 0.67) excluded
        assert row2["ann_basis_mean"] == pytest.approx(0.05)


class TestFailClosed:
    def test_insufficient_overlap_is_error_row(self) -> None:
        spot, future = _planted_carry(days=4, delivery=DELIVERY)
        _frame_out, receipt = run_basis_carry(
            spot=spot,
            spot_label="SYNTHETIC",
            contracts=[
                CarryContractInput(
                    symbol="FI_XBTUSD_250926",
                    frame=future,
                    delivery=DELIVERY,
                    data_label="SYNTHETIC",
                )
            ],
        )
        row = receipt["results"][0]
        assert row["status"] == "error"
        assert "insufficient_overlap" in row["error"]
        assert receipt["n_error_rows"] == 1

    def test_non_monotonic_contract_frame_is_error_row(self) -> None:
        spot, _future = _planted_carry(delivery=DELIVERY)
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(10)]
        dates[3], dates[4] = dates[4], dates[3]  # out of order
        future = _frame(dates, [100.0] * 10)
        _frame_out, receipt = run_basis_carry(
            spot=spot,
            spot_label="SYNTHETIC",
            contracts=[
                CarryContractInput(
                    symbol="FI_XBTUSD_250926",
                    frame=future,
                    delivery=DELIVERY,
                    data_label="SYNTHETIC",
                )
            ],
        )
        row = receipt["results"][0]
        assert row["status"] == "error"
        assert "strictly increasing" in row["error"]

    def test_unparseable_delivery_is_error_row(self) -> None:
        spot, future = _planted_carry(delivery=DELIVERY)
        _frame_out, receipt = run_basis_carry(
            spot=spot,
            spot_label="SYNTHETIC",
            contracts=[
                CarryContractInput(
                    symbol="XX_BAD_1",
                    frame=future,
                    delivery="not-a-date",
                    data_label="SYNTHETIC",
                )
            ],
        )
        row = receipt["results"][0]
        assert row["status"] == "error"
        assert "unparseable" in row["error"]

    def test_non_monotonic_spot_raises(self) -> None:
        _spot, future = _planted_carry(delivery=DELIVERY)
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(10)]
        dates[2], dates[5] = dates[5], dates[2]
        spot = _frame(dates, [100.0] * 10)
        with pytest.raises(ValueError, match="strictly increasing"):
            run_basis_carry(
                spot=spot,
                spot_label="SYNTHETIC",
                contracts=[
                    CarryContractInput(
                        symbol="FI_XBTUSD_250926",
                        frame=future,
                        delivery=DELIVERY,
                        data_label="SYNTHETIC",
                    )
                ],
            )

    def test_empty_spot_raises(self) -> None:
        _spot, future = _planted_carry(delivery=DELIVERY)
        with pytest.raises(ValueError, match="frame is empty"):
            run_basis_carry(
                spot=_frame([], []),
                spot_label="SYNTHETIC",
                contracts=[
                    CarryContractInput(
                        symbol="FI_XBTUSD_250926",
                        frame=future,
                        delivery=DELIVERY,
                        data_label="SYNTHETIC",
                    )
                ],
            )

    def test_mixed_data_labels_fail_closed(self) -> None:
        spot, future = _planted_carry(delivery=DELIVERY)
        with pytest.raises(ValueError, match="mixed data_label"):
            run_basis_carry(
                spot=spot,
                spot_label="SYNTHETIC",
                contracts=[
                    CarryContractInput(
                        symbol="FI_XBTUSD_250926",
                        frame=future,
                        delivery=DELIVERY,
                        data_label="kraken",
                    )
                ],
            )

    def test_no_contracts_raises(self) -> None:
        spot, _future = _planted_carry(delivery=DELIVERY)
        with pytest.raises(ValueError, match="at least one contract"):
            run_basis_carry(spot=spot, spot_label="SYNTHETIC", contracts=[])


class TestReceipt:
    def test_receipt_seals_and_verifies(self, tmp_path: Path) -> None:
        _frame_out, receipt = _run()
        assert basis_carry_contract_errors(receipt) == []
        path = write_basis_carry_receipt(receipt, tmp_path)
        from quant_fund.research.receipt_v2 import verify_receipt_file

        result = verify_receipt_file(path)
        assert result["valid"], result["errors"]
        payload = json.loads(path.read_text())
        assert payload["schema"] == "basis_carry.v1"
        assert payload["data_label"] == "SYNTHETIC"
        assert payload["live_pnl_claim"] is False

    def test_receipt_v2_envelope_verifies(self, tmp_path: Path) -> None:
        _frame_out, receipt = _run()
        path = write_basis_carry_receipt(receipt, tmp_path, receipt_version=2)
        from quant_fund.research.receipt_v2 import verify_receipt_file

        result = verify_receipt_file(path)
        assert result["valid"], result["errors"]
        payload = json.loads(path.read_text())
        assert payload["schema"] == "receipt.v2"
        assert payload["kind"] == "basis_carry"
        assert payload["payload"]["schema"] == "basis_carry.v1"

    def test_contract_dispatch_via_lane_contracts(self) -> None:
        _frame_out, receipt = _run()
        from quant_fund.research.lane_contracts import lane_contract_errors

        assert lane_contract_errors(receipt) == []

    def test_forged_residual_fails_contract(self) -> None:
        _frame_out, receipt = _run()
        receipt["results"][0]["convergence_residual"] = 0.9
        assert basis_carry_contract_errors(receipt) != []

    def test_forged_inputs_digest_fails_contract(self) -> None:
        _frame_out, receipt = _run()
        receipt["inputs"]["spot"]["sha256"] = "0" * 64
        assert "inputs_sha256_mismatch" in basis_carry_contract_errors(receipt)

    def test_forbidden_metric_tokens_absent(self) -> None:
        from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

        _frame_out, receipt = _run()
        assert family_blob_forbidden_metrics_absent(receipt)


def _kraken_ohlc_payload(rows: list[list[Any]], pair_key: str = "XXBTZUSD") -> dict:
    return {"error": [], "result": {pair_key: rows, "last": 1_700_086_400}}


def _spot_row(open_s: int, price: str = "100") -> list[Any]:
    return [open_s, price, "110", "95", price, "100.5", "1.5", 10]


def _mark_candle(open_ms: int, price: str = "100") -> dict[str, Any]:
    return {
        "time": open_ms,
        "open": price,
        "high": "110",
        "low": "95",
        "close": price,
        "volume": "0",
    }


class _Client:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.urls: list[str] = []

    def get_json(self, url: str, headers: dict | None = None) -> Any:  # noqa: ARG002
        self.urls.append(url)
        return self._payload


class TestKrakenAdapters:
    def test_spot_ohlc_normalizes_and_drops_forming_candle(self) -> None:
        from quant_fund.data.sources.adapters import KrakenSpotOhlcSource
        from quant_fund.data.sources.base import utc_now

        now_s = int(utc_now().timestamp())
        rows = [
            _spot_row(now_s - 3 * 86_400),
            _spot_row(now_s - 2 * 86_400),
            _spot_row(now_s - 3_600),  # still forming — close is in the future
        ]
        src = KrakenSpotOhlcSource(client=_Client(_kraken_ohlc_payload(rows)))
        frame = src.fetch(pair="XBTUSD", interval=1440)
        assert frame.height == 2
        assert frame["security_id"].to_list() == ["XBTUSD"] * 2
        assert frame["close"].to_list() == [100.0, 100.0]
        # Only the forming candle -> fail closed.
        src2 = KrakenSpotOhlcSource(client=_Client(_kraken_ohlc_payload([rows[2]])))
        with pytest.raises(SourceError, match="in-progress"):
            src2.fetch(pair="XBTUSD", interval=1440)

    def test_spot_ohlc_fail_closed_cases(self) -> None:
        from quant_fund.data.sources.adapters import KrakenSpotOhlcSource

        # API error list is honored.
        src = KrakenSpotOhlcSource(
            client=_Client({"error": ["EGeneral:Unknown asset"], "result": {}})
        )
        with pytest.raises(SourceError, match="API error"):
            src.fetch(pair="BADPAIR")
        # Two pair series is ambiguous -> fail closed.
        payload = _kraken_ohlc_payload([_spot_row(1_700_000_000)])
        payload["result"]["XETHZUSD"] = [_spot_row(1_700_000_000)]
        src = KrakenSpotOhlcSource(client=_Client(payload))
        with pytest.raises(SourceError, match="pair series"):
            src.fetch()
        # Malformed row shape fails closed.
        bad = _kraken_ohlc_payload([[1_700_000_000, "100"]])
        src = KrakenSpotOhlcSource(client=_Client(bad))
        with pytest.raises(SourceError, match="malformed"):
            src.fetch()
        # Bad interval rejected before any call.
        with pytest.raises(ValueError, match="interval"):
            src.fetch(interval=3)

    def test_futures_mark_normalizes_and_drops_forming_candle(self) -> None:
        from quant_fund.data.sources.adapters import KrakenFuturesMarkSource
        from quant_fund.data.sources.base import utc_now

        now_ms = int(utc_now().timestamp() * 1000)
        candles = [
            _mark_candle(now_ms - 3 * 86_400_000, "100"),
            _mark_candle(now_ms - 2 * 86_400_000, "101"),
            _mark_candle(now_ms - 3_600_000, "102"),  # still forming
        ]
        client = _Client({"candles": candles})
        src = KrakenFuturesMarkSource(client=client)
        frame = src.fetch(symbol="FI_XBTUSD_261225")
        assert frame.height == 2
        assert "mark/FI_XBTUSD_261225/1d" in client.urls[0]
        # Empty candle list fails closed.
        src = KrakenFuturesMarkSource(client=_Client({"candles": []}))
        with pytest.raises(SourceError, match="no candles"):
            src.fetch(symbol="FI_XBTUSD_261225")
        # Malformed candle shape fails closed.
        src = KrakenFuturesMarkSource(client=_Client({"candles": [{"close": "1"}]}))
        with pytest.raises(SourceError, match="malformed"):
            src.fetch(symbol="FI_XBTUSD_261225")
        # Malformed contract symbol rejected before any call.
        with pytest.raises(ValueError, match="contract symbols"):
            src.fetch(symbol="not a symbol")

    def test_futures_universe_carries_delivery_and_filters(self) -> None:
        from quant_fund.data.sources.adapters import KrakenFuturesUniverseSource

        payload = {
            "result": "success",
            "serverTime": "2026-09-29T12:00:00.000Z",
            "instruments": [
                {
                    "symbol": "FI_XBTUSD_261225",
                    "type": "futures_inverse",
                    "underlying": "rr_xbtusd",
                    "lastTradingTime": "2026-12-25T16:00:00Z",
                    "tradeable": True,
                },
                {
                    "symbol": "PI_XBTUSD",  # perpetual — no lastTradingTime
                    "type": "futures_inverse",
                    "underlying": "rr_xbtusd",
                    "tradeable": True,
                },
            ],
        }
        src = KrakenFuturesUniverseSource(client=_Client(payload))
        frame = src.fetch()
        assert frame["security_id"].to_list() == ["FI_XBTUSD_261225"]
        assert frame["last_trading_ms"].to_list() == [
            int(datetime(2026, 12, 25, 16, 0, tzinfo=UTC).timestamp() * 1000)
        ]
        frame_all = src.fetch(include_undated=True)
        assert frame_all.height == 2
        assert frame_all.filter(pl.col("security_id") == "PI_XBTUSD")[
            "last_trading_ms"
        ].to_list() == [None]
        # Schema drift fails closed.
        src = KrakenFuturesUniverseSource(client=_Client({"result": "success"}))
        with pytest.raises(SourceError, match="instruments list"):
            src.fetch()
        src = KrakenFuturesUniverseSource(
            client=_Client({"instruments": [{"type": "futures_inverse"}]})
        )
        with pytest.raises(SourceError, match="required fields"):
            src.fetch()

    def test_kraken_sources_registered(self) -> None:
        from quant_fund.data.sources.registry import get_source, source_names

        names = source_names()
        assert "kraken_spot" in names
        assert "kraken_futures_mark" in names
        assert "kraken_futures_universe" in names
        assert get_source("kraken").name == "kraken_spot"
        assert get_source("kraken_fut").name == "kraken_futures_mark"
        assert get_source("kraken_fut_universe").name == "kraken_futures_universe"


def test_basis_carry_cli_writes_receipt(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    spot, future = _planted_carry(days=30, delivery=DELIVERY)
    spot_path = tmp_path / "spot.parquet"
    fut_path = tmp_path / "fut.parquet"
    spot.write_parquet(spot_path)
    future.write_parquet(fut_path)
    out_dir = tmp_path / "receipts"
    result = CliRunner().invoke(
        app,
        [
            "basis-carry",
            "--config",
            "configs/research.yaml",
            "--spot",
            str(spot_path),
            "--contract-path",
            f"FI_XBTUSD_250926={fut_path}",
            "--data-label",
            "SYNTHETIC",
            "--out-dir",
            str(out_dir),
            "--strict",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    receipts = list(out_dir.glob("basis_carry_*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["kind"] == "basis_carry"
    assert payload["results"][0]["status"] == "ok"


def test_basis_carry_cli_strict_exits_on_error_row(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    spot, _future = _planted_carry(days=30, delivery=DELIVERY)
    # Contract frame overlaps for fewer than min_overlap dates.
    future = _frame([date(2030, 1, 1) + timedelta(days=i) for i in range(30)], [1.0] * 30)
    spot_path = tmp_path / "spot.parquet"
    fut_path = tmp_path / "fut.parquet"
    spot.write_parquet(spot_path)
    future.write_parquet(fut_path)
    result = CliRunner().invoke(
        app,
        [
            "basis-carry",
            "--config",
            "configs/research.yaml",
            "--spot",
            str(spot_path),
            "--contract-path",
            f"FI_XBTUSD_250926={fut_path}",
            "--data-label",
            "SYNTHETIC",
            "--out-dir",
            str(tmp_path / "receipts"),
            "--strict",
        ],
    )
    assert result.exit_code == 1, result.output
