"""crossvenue_basis: cross-venue funding/basis bench (P5.5).

Covers the OKX and Kraken-funding source adapters with canned fixtures
(in-progress candle dropped via OKX's confirm flag, malformed payloads fail
closed, pagination walks backward), the bench's known-answer path on
planted basis/funding differentials, fail-closed input cases, the sealed
``crossvenue_basis.v1`` receipt contract, and the CLI surface. All frames
here are SYNTHETIC fixtures — correctness evidence, never market evidence.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant_fund.data.sources.base import SourceError
from quant_fund.research.crossvenue_basis import (
    VenueLeg,
    crossvenue_basis_contract_errors,
    run_crossvenue_basis,
    write_crossvenue_basis_receipt,
)

pytestmark = pytest.mark.synthetic


def _price_frame(dates: list[date], closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"event_time": dates, "close": closes})


def _funding_frame(stamps: list[datetime], rates: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"event_time": stamps, "value": rates})


START = date(2025, 1, 1)


def _leg(
    venue: str,
    *,
    days: int = 30,
    basis_level: float,
    daily_funding: float | None = 0.0003,
    settlements_per_day: int = 3,
    label: str = "SYNTHETIC",
) -> VenueLeg:
    """Planted leg: flat spot 100, mark at constant basis, constant funding."""
    dates = [START + timedelta(days=i) for i in range(days)]
    spot = _price_frame(dates, [100.0] * days)
    mark = _price_frame(dates, [100.0 * (1.0 + basis_level)] * days)
    funding = None
    if daily_funding is not None:
        stamps: list[datetime] = []
        rates: list[float] = []
        step = 24 // settlements_per_day
        for day in dates:
            for k in range(settlements_per_day):
                stamps.append(datetime.combine(day, time(step * k, 0), tzinfo=UTC))
                rates.append(daily_funding / settlements_per_day)
        funding = _funding_frame(stamps, rates)
    return VenueLeg(venue=venue, spot=spot, mark=mark, funding=funding, data_label=label)


def _run(**kwargs: Any):
    legs = [
        _leg("kraken", basis_level=0.001, daily_funding=0.0003),
        _leg("okx", basis_level=0.003, daily_funding=0.0001),
    ]
    return run_crossvenue_basis(legs=legs, asset="BTC", **kwargs)


class TestKnownAnswer:
    def test_constant_differentials_recover_exactly(self) -> None:
        frame, receipt = _run()
        pair = receipt["results"][0]
        assert pair["status"] == "ok"
        assert pair["pair"] == "kraken~okx"
        # Planted: basis_kraken=0.001, basis_okx=0.003 -> diff -0.002 flat.
        assert pair["basis_diff"]["mean"] == pytest.approx(-0.002)
        assert pair["basis_diff"]["std"] == pytest.approx(0.0, abs=1e-12)
        assert pair["basis_diff"]["n_dates"] == 30
        assert pair["basis_diff"]["sign_flips"] == 0
        # Funding: kraken 3e-4/day vs okx 1e-4/day -> +2e-4/day flat.
        assert pair["funding_diff"]["status"] == "ok"
        assert pair["funding_diff"]["mean"] == pytest.approx(0.0002)
        assert pair["funding_diff"]["n_dates"] == 30
        assert frame.height == 1
        assert frame["basis_diff_mean"].to_list() == pytest.approx([-0.002])
        legs = {row["venue"]: row for row in receipt["legs"]}
        assert legs["kraken"]["basis_mean"] == pytest.approx(0.001)
        assert legs["okx"]["basis_mean"] == pytest.approx(0.003)
        assert legs["kraken"]["funding"]["daily_mean"] == pytest.approx(0.0003)
        assert legs["kraken"]["funding"]["cadence_seconds"] == pytest.approx(28800.0)
        assert legs["okx"]["funding"]["expected_settlements_per_day"] == pytest.approx(3.0)

    def test_sign_flips_and_frac_positive_counted(self) -> None:
        dates = [START + timedelta(days=i) for i in range(10)]
        spot = _price_frame(dates, [100.0] * 10)
        mark_a = _price_frame(dates, [100.0 + i for i in range(10)])
        mark_b = _price_frame(dates, [100.5] * 10)
        legs = [
            VenueLeg(venue="avenu", spot=spot, mark=mark_a, funding=None, data_label="SYNTHETIC"),
            VenueLeg(venue="bvenu", spot=spot, mark=mark_b, funding=None, data_label="SYNTHETIC"),
        ]
        _frame, receipt = run_crossvenue_basis(legs=legs, asset="BTC", min_overlap=5)
        pair = receipt["results"][0]
        # basis_a - basis_b = (i - 0.5)/100: negative at i=0, positive after.
        assert pair["basis_diff"]["frac_positive"] == pytest.approx(0.9)
        assert pair["basis_diff"]["sign_flips"] == 1
        assert pair["funding_diff"]["status"] == "unavailable"
        assert pair["basis_diff"]["mean"] == pytest.approx(
            sum((i - 0.5) / 100.0 for i in range(10)) / 10
        )


class TestFailClosed:
    def test_single_leg_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            run_crossvenue_basis(legs=[_leg("kraken", basis_level=0.001)], asset="BTC")

    def test_duplicate_venues_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate venue"):
            run_crossvenue_basis(
                legs=[_leg("kraken", basis_level=0.001), _leg("kraken", basis_level=0.002)],
                asset="BTC",
            )

    def test_mixed_synthetic_and_live_labels_fail_closed(self) -> None:
        legs = [
            _leg("kraken", basis_level=0.001, label="SYNTHETIC"),
            _leg("okx", basis_level=0.003, label="okx"),
        ]
        with pytest.raises(ValueError, match="SYNTHETIC"):
            run_crossvenue_basis(legs=legs, asset="BTC")

    def test_two_live_venues_join_labels(self) -> None:
        legs = [
            _leg("kraken", basis_level=0.001, label="kraken"),
            _leg("okx", basis_level=0.003, label="okx"),
        ]
        _frame, receipt = run_crossvenue_basis(legs=legs, asset="BTC")
        assert receipt["data_label"] == "kraken+okx"

    def test_insufficient_leg_overlap_is_error_row(self) -> None:
        legs = [
            _leg("kraken", basis_level=0.001),
            _leg(
                "okx",
                days=3,
                basis_level=0.003,
            ),
        ]
        _frame, receipt = run_crossvenue_basis(legs=legs, asset="BTC")
        okx_leg = next(row for row in receipt["legs"] if row["venue"] == "okx")
        assert okx_leg["status"] == "error"
        assert "insufficient_overlap" in okx_leg["error"]
        assert receipt["results"] == [] or all(row["status"] != "ok" for row in receipt["results"])
        assert receipt["n_error_rows"] >= 1

    def test_insufficient_pair_overlap_is_error_row(self) -> None:
        legs = [
            _leg("kraken", basis_level=0.001),
            VenueLeg(
                venue="okx",
                spot=_price_frame(
                    [date(2030, 1, 1) + timedelta(days=i) for i in range(10)], [100.0] * 10
                ),
                mark=_price_frame(
                    [date(2030, 1, 1) + timedelta(days=i) for i in range(10)], [100.3] * 10
                ),
                funding=None,
                data_label="SYNTHETIC",
            ),
        ]
        _frame, receipt = run_crossvenue_basis(legs=legs, asset="BTC")
        pair = receipt["results"][0]
        assert pair["status"] == "error"
        assert "insufficient_pair_overlap" in pair["error"]

    def test_non_monotonic_spot_is_error_leg(self) -> None:
        dates = [START + timedelta(days=i) for i in range(10)]
        dates[2], dates[5] = dates[5], dates[2]
        legs = [
            VenueLeg(
                venue="kraken",
                spot=_price_frame(dates, [100.0] * 10),
                mark=_price_frame([START + timedelta(days=i) for i in range(10)], [100.1] * 10),
                funding=None,
                data_label="SYNTHETIC",
            ),
            _leg("okx", basis_level=0.003),
        ]
        _frame, receipt = run_crossvenue_basis(legs=legs, asset="BTC")
        kraken_leg = next(row for row in receipt["legs"] if row["venue"] == "kraken")
        assert kraken_leg["status"] == "error"
        assert "strictly increasing" in kraken_leg["error"]

    def test_non_monotonic_funding_is_error_leg(self) -> None:
        bad = _leg("kraken", basis_level=0.001)
        stamps = [datetime(2025, 1, 1, 8, tzinfo=UTC), datetime(2025, 1, 1, 7, tzinfo=UTC)]
        bad = VenueLeg(
            venue="kraken",
            spot=bad.spot,
            mark=bad.mark,
            funding=_funding_frame(stamps, [1e-4, 1e-4]),
            data_label="SYNTHETIC",
        )
        _frame, receipt = run_crossvenue_basis(
            legs=[bad, _leg("okx", basis_level=0.003)], asset="BTC"
        )
        kraken_leg = next(row for row in receipt["legs"] if row["venue"] == "kraken")
        assert kraken_leg["status"] == "error"
        assert "funding" in kraken_leg["error"]

    def test_missing_close_column_is_error_leg(self) -> None:
        legs = [
            VenueLeg(
                venue="kraken",
                spot=pl.DataFrame({"event_time": [START], "price": [100.0]}),
                mark=_leg("tmp", basis_level=0.0).mark,
                funding=None,
                data_label="SYNTHETIC",
            ),
            _leg("okx", basis_level=0.003),
        ]
        _frame, receipt = run_crossvenue_basis(legs=legs, asset="BTC")
        kraken_leg = next(row for row in receipt["legs"] if row["venue"] == "kraken")
        assert kraken_leg["status"] == "error"
        assert "close" in kraken_leg["error"]


class TestReceipt:
    def test_receipt_seals_and_verifies(self, tmp_path: Path) -> None:
        _frame, receipt = _run()
        assert crossvenue_basis_contract_errors(receipt) == []
        path = write_crossvenue_basis_receipt(receipt, tmp_path)
        from quant_fund.research.receipt_v2 import verify_receipt_file

        result = verify_receipt_file(path)
        assert result["valid"], result["errors"]
        payload = json.loads(path.read_text())
        assert payload["schema"] == "crossvenue_basis.v1"
        assert payload["data_label"] == "SYNTHETIC"
        assert payload["live_pnl_claim"] is False

    def test_receipt_v2_envelope_verifies(self, tmp_path: Path) -> None:
        _frame, receipt = _run()
        path = write_crossvenue_basis_receipt(receipt, tmp_path, receipt_version=2)
        from quant_fund.research.receipt_v2 import verify_receipt_file

        result = verify_receipt_file(path)
        assert result["valid"], result["errors"]
        payload = json.loads(path.read_text())
        assert payload["schema"] == "receipt.v2"
        assert payload["kind"] == "crossvenue_basis"
        assert payload["payload"]["schema"] == "crossvenue_basis.v1"

    def test_contract_dispatch_via_lane_contracts(self) -> None:
        _frame, receipt = _run()
        from quant_fund.research.lane_contracts import lane_contract_errors

        assert lane_contract_errors(receipt) == []

    def test_forged_basis_diff_fails_contract(self) -> None:
        _frame, receipt = _run()
        receipt["results"][0]["basis_diff"]["mean"] = 0.9
        assert crossvenue_basis_contract_errors(receipt) != []

    def test_forged_leg_series_fails_contract(self) -> None:
        _frame, receipt = _run()
        receipt["legs"][0]["basis_series"][5][1] = 9.9
        assert crossvenue_basis_contract_errors(receipt) != []

    def test_forged_inputs_digest_fails_contract(self) -> None:
        _frame, receipt = _run()
        receipt["inputs"]["legs"]["kraken"]["spot"]["sha256"] = "0" * 64
        assert "inputs_sha256_mismatch" in crossvenue_basis_contract_errors(receipt)

    def test_forbidden_metric_tokens_absent(self) -> None:
        from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

        _frame, receipt = _run()
        assert family_blob_forbidden_metrics_absent(receipt)


class _Client:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.urls: list[str] = []

    def get_json(self, url: str, headers: dict | None = None) -> Any:  # noqa: ARG002
        self.urls.append(url)
        return self._payload


class _PagedClient:
    """Serves sequential canned pages for pagination tests."""

    def __init__(self, pages: list[Any]) -> None:
        self._pages = list(pages)
        self.urls: list[str] = []

    def get_json(self, url: str, headers: dict | None = None) -> Any:  # noqa: ARG002
        self.urls.append(url)
        return self._pages.pop(0) if self._pages else {"code": "0", "data": [], "msg": ""}


def _okx_candle(ts_ms: int, close: str = "100", confirm: str = "1", fields: int = 9) -> list[Any]:
    row: list[Any] = [str(ts_ms), "99", "110", "95", close]
    if fields == 9:
        row += ["10", "1000", "1000"]
    row.append(confirm)
    return row


def _okx_envelope(data: Any, code: str = "0") -> dict[str, Any]:
    return {"code": code, "data": data, "msg": ""}


class TestOkxAdapters:
    def test_spot_normalizes_and_drops_unconfirmed(self) -> None:
        from quant_fund.data.sources.adapters import OkxSpotOhlcSource
        from quant_fund.data.sources.base import utc_now

        now_ms = int(utc_now().timestamp() * 1000)
        day = 86_400_000
        rows = [
            _okx_candle(now_ms - 4 * day),  # newest page order: newest first
            _okx_candle(now_ms - 3 * day),
            _okx_candle(now_ms - 2 * day),
            _okx_candle(now_ms - 3600_000, confirm="0"),  # still forming
        ]
        src = OkxSpotOhlcSource(client=_Client(_okx_envelope(rows)))
        frame = src.fetch(inst_id="BTC-USDT", bar="1Dutc")
        assert frame.height == 3
        assert frame["security_id"].to_list() == ["BTC-USDT"] * 3
        assert frame["event_time"].is_sorted()
        src2 = OkxSpotOhlcSource(client=_Client(_okx_envelope([rows[3]])))
        with pytest.raises(SourceError, match="in-progress"):
            src2.fetch(inst_id="BTC-USDT")

    def test_spot_fail_closed_cases(self) -> None:
        from quant_fund.data.sources.adapters import OkxSpotOhlcSource

        src = OkxSpotOhlcSource(client=_Client(_okx_envelope([], code="60011")))
        with pytest.raises(SourceError, match="API error"):
            src.fetch(inst_id="BTC-USDT")
        src = OkxSpotOhlcSource(client=_Client(_okx_envelope([])))
        with pytest.raises(SourceError, match="no rows"):
            src.fetch(inst_id="BTC-USDT")
        src = OkxSpotOhlcSource(client=_Client(_okx_envelope([["bad-ts"]])))
        with pytest.raises(SourceError, match="malformed"):
            src.fetch(inst_id="BTC-USDT")
        with pytest.raises(ValueError, match="instrument ids"):
            src.fetch(inst_id="not an id")
        with pytest.raises(ValueError, match="bar"):
            src.fetch(inst_id="BTC-USDT", bar="13m")

    def test_spot_paginates_backward(self) -> None:
        from quant_fund.data.sources.adapters import OkxSpotOhlcSource

        day = 86_400_000
        base = 1_700_000_000_000
        page1 = [_okx_candle(base + 2 * day), _okx_candle(base + day)]
        page2 = [_okx_candle(base)]  # short page ends the walk
        client = _PagedClient([_okx_envelope(page1), _okx_envelope(page2)])
        src = OkxSpotOhlcSource(client=client)
        frame = src.fetch(inst_id="BTC-USDT", limit=2, max_pages=5, pause_seconds=0)
        assert frame.height == 3
        assert f"after={base + day}" in client.urls[1]

    def test_mark_candles_use_zero_volume(self) -> None:
        from quant_fund.data.sources.adapters import OkxMarkCandlesSource

        day = 86_400_000
        base = 1_700_000_000_000
        rows = [_okx_candle(base, fields=6), _okx_candle(base + day, fields=6, confirm="0")]
        src = OkxMarkCandlesSource(client=_Client(_okx_envelope(rows)))
        frame = src.fetch(inst_id="BTC-USDT-SWAP")
        assert frame.height == 1
        assert frame["volume"].to_list() == [0.0]
        assert "mark-price-candles" in src.client.urls[0]

    def test_funding_normalizes_and_drops_future(self) -> None:
        from quant_fund.data.sources.adapters import OkxFundingHistorySource
        from quant_fund.data.sources.base import utc_now

        now_ms = int(utc_now().timestamp() * 1000)
        step = 8 * 3_600_000
        data = [
            {
                "instId": "BTC-USDT-SWAP",
                "instType": "SWAP",
                "fundingTime": str(now_ms + step),  # not yet charged
                "fundingRate": "0.0001",
                "realizedRate": "0.0001",
            },
            {
                "instId": "BTC-USDT-SWAP",
                "instType": "SWAP",
                "fundingTime": str(now_ms - step),
                "fundingRate": "-0.0002",
                "realizedRate": "-0.0002",
            },
            {
                "instId": "BTC-USDT-SWAP",
                "instType": "SWAP",
                "fundingTime": str(now_ms - 2 * step),
                "fundingRate": "0.0003",
                "realizedRate": "0.0003",
            },
        ]
        src = OkxFundingHistorySource(client=_Client(_okx_envelope(data)))
        frame = src.fetch(inst_id="BTC-USDT-SWAP")
        assert frame.height == 2  # future fundingTime dropped
        assert frame["value"].to_list() == [0.0003, -0.0002]
        src = OkxFundingHistorySource(client=_Client(_okx_envelope([])))
        with pytest.raises(SourceError, match="no rows"):
            src.fetch(inst_id="BTC-USDT-SWAP")
        src = OkxFundingHistorySource(client=_Client(_okx_envelope([{"instId": "X"}])))
        with pytest.raises(SourceError, match="malformed"):
            src.fetch(inst_id="BTC-USDT-SWAP")

    def test_swap_universe_filters_and_carries_metadata(self) -> None:
        from quant_fund.data.sources.adapters import OkxSwapUniverseSource

        payload = _okx_envelope(
            [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instFamily": "BTC-USDT",
                    "state": "live",
                    "ctType": "linear",
                    "ctVal": "0.01",
                    "ctValCcy": "BTC",
                    "settleCcy": "USDT",
                    "tickSz": "0.1",
                    "lotSz": "0.01",
                    "listTime": "1573557408000",
                },
                {
                    "instId": "DEAD-USDT-SWAP",
                    "instFamily": "DEAD-USDT",
                    "state": "preopen",
                    "ctType": "linear",
                },
            ]
        )
        src = OkxSwapUniverseSource(client=_Client(payload))
        frame = src.fetch()
        assert frame["security_id"].to_list() == ["BTC-USDT-SWAP"]
        assert frame["ct_val_ccy"].to_list() == ["BTC"]
        frame_all = src.fetch(live_only=False)
        assert frame_all.height == 2
        src = OkxSwapUniverseSource(client=_Client(_okx_envelope([{"state": "live"}])))
        with pytest.raises(SourceError, match="required fields"):
            src.fetch()

    def test_kraken_funding_normalizes(self) -> None:
        from quant_fund.data.sources.adapters import KrakenFuturesFundingSource

        payload = {
            "result": "success",
            "serverTime": "2025-09-25T00:00:00Z",
            "rates": [
                {
                    "timestamp": "2025-09-24T08:00:00Z",
                    "fundingRate": 1.5,
                    "relativeFundingRate": "0.000011795815277778",
                },
                {
                    "timestamp": "2025-09-24T09:00:00Z",
                    "fundingRate": -0.5,
                    "relativeFundingRate": "-0.000005",
                },
            ],
        }
        src = KrakenFuturesFundingSource(client=_Client(payload))
        frame = src.fetch(symbol="PF_XBTUSD")
        assert frame.height == 2
        assert frame["value"].to_list() == pytest.approx([1.1795815277778e-05, -5e-06])
        src = KrakenFuturesFundingSource(client=_Client({"result": "error", "errors": [{}]}))
        with pytest.raises(SourceError, match="API error"):
            src.fetch(symbol="PF_XBTUSD")
        src = KrakenFuturesFundingSource(client=_Client({"result": "success", "rates": []}))
        with pytest.raises(SourceError, match="no rates"):
            src.fetch(symbol="PF_XBTUSD")
        with pytest.raises(ValueError, match="contract symbols"):
            src.fetch(symbol="bogus")

    def test_sources_registered(self) -> None:
        from quant_fund.data.sources.registry import get_source, source_names

        names = source_names()
        for name in (
            "okx_spot",
            "okx_mark",
            "okx_funding",
            "okx_swap_universe",
            "kraken_funding",
        ):
            assert name in names
        assert get_source("okx").name == "okx_spot"
        assert get_source("okx_fund").name == "okx_funding"
        assert get_source("kraken_fund").name == "kraken_funding"


def _file_legs(tmp_path: Path) -> list[str]:
    """Write SYNTHETIC leg frames; return the two --leg spec strings."""
    specs = []
    for venue, basis in (("va", 0.001), ("vb", 0.003)):
        dates = [START + timedelta(days=i) for i in range(10)]
        spot = _price_frame(dates, [100.0] * 10)
        mark = _price_frame(dates, [100.0 * (1 + basis)] * 10)
        spot_path = tmp_path / f"{venue}_spot.parquet"
        mark_path = tmp_path / f"{venue}_mark.parquet"
        spot.write_parquet(spot_path)
        mark.write_parquet(mark_path)
        specs.append(f"venue={venue},spot={spot_path},mark={mark_path}")
    return specs


def test_xvenue_basis_cli_writes_receipt(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    specs = _file_legs(tmp_path)
    out_dir = tmp_path / "receipts"
    args = [
        "xvenue-basis",
        "--config",
        "configs/research.yaml",
        "--data-label",
        "SYNTHETIC",
        "--out-dir",
        str(out_dir),
        "--strict",
    ]
    for spec in specs:
        args += ["--leg", spec]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    receipts = list(out_dir.glob("crossvenue_basis_*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text())
    assert payload["kind"] == "crossvenue_basis"
    assert payload["results"][0]["status"] == "ok"


def test_xvenue_basis_cli_strict_exits_on_error_leg(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    specs = _file_legs(tmp_path)
    # Second leg's mark overlaps nothing in its spot window.
    dates_far = [date(2031, 1, 1) + timedelta(days=i) for i in range(10)]
    far_path = tmp_path / "vb_mark.parquet"
    _price_frame(dates_far, [100.0] * 10).write_parquet(far_path)
    specs[1] = f"venue=vb,spot={tmp_path / 'vb_spot.parquet'},mark={far_path}"
    args = [
        "xvenue-basis",
        "--config",
        "configs/research.yaml",
        "--data-label",
        "SYNTHETIC",
        "--out-dir",
        str(tmp_path / "receipts"),
        "--strict",
    ]
    for spec in specs:
        args += ["--leg", spec]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output


def test_xvenue_basis_cli_rejects_single_leg(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    specs = _file_legs(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "xvenue-basis",
            "--leg",
            specs[0],
            "--data-label",
            "SYNTHETIC",
            "--out-dir",
            str(tmp_path / "receipts"),
        ],
    )
    assert result.exit_code != 0
    assert ">= 2 legs" in result.output or "two venue legs" in result.output


@pytest.mark.network
def test_xvenue_live_drill(tmp_path: Path) -> None:
    """Real pull: Kraken + OKX BTC legs produce a sealed receipt.

    Drill evidence only — asserts the adapters fetch real frames and the
    bench computes its contract; it makes no market claim about the values.
    """
    from quant_fund.data.collector import collect_source
    from quant_fund.research.crossvenue_basis import crossvenue_basis_verdict

    legs = [
        VenueLeg(
            venue="kraken",
            spot=collect_source(
                "kraken_spot", tmp_path, fetch_kwargs={"pair": "XBTUSD", "interval": 1440}
            ).frame,
            mark=collect_source(
                "kraken_futures_mark", tmp_path, fetch_kwargs={"symbol": "PF_XBTUSD"}
            ).frame,
            funding=collect_source(
                "kraken_funding", tmp_path, fetch_kwargs={"symbol": "PF_XBTUSD"}
            ).frame,
            data_label="kraken",
        ),
        VenueLeg(
            venue="okx",
            spot=collect_source(
                "okx_spot", tmp_path, fetch_kwargs={"inst_id": "BTC-USDT", "bar": "1Dutc"}
            ).frame,
            mark=collect_source(
                "okx_mark", tmp_path, fetch_kwargs={"inst_id": "BTC-USDT-SWAP", "bar": "1Dutc"}
            ).frame,
            funding=collect_source(
                "okx_funding", tmp_path, fetch_kwargs={"inst_id": "BTC-USDT-SWAP"}
            ).frame,
            data_label="okx",
        ),
    ]
    frame, receipt = run_crossvenue_basis(legs=legs, asset="BTC")
    assert receipt["data_label"] == "kraken+okx"
    assert crossvenue_basis_contract_errors(receipt) == []
    assert crossvenue_basis_verdict(receipt) == "pass"
    pair = receipt["results"][0]
    assert pair["basis_diff"]["n_dates"] >= 30
    assert pair["funding_diff"]["status"] == "ok"
    assert frame.height == 1
    path = write_crossvenue_basis_receipt(receipt, tmp_path / "receipts")
    from quant_fund.research.receipt_v2 import verify_receipt_file

    assert verify_receipt_file(path)["valid"]
