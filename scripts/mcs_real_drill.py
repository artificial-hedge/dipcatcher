"""MCS drill on REAL tape: sequential confidence set over vol forecasters.

Loads the collected Yahoo EOD bars (``data/file_us_wide/bronze/bars.parquet``),
builds a daily-RV vol shard for a liquid symbol, scores every vol forecaster
origin-by-origin with proper QLIKE losses, and feeds the per-origin loss
streams into ``AnytimeMCS``. Writes a sealed ``receipts/mcs_real_drill.json``
stamped ``data_label=yahoo_eod`` — a real-data sequential verdict, not a
synthetic correctness check. Proper scores only; no P&L claims.

Usage: ``python -m scripts.mcs_real_drill [bars.parquet]`` — the path
    defaults to the repo-local collected tape and accepts an override since
    ``data/`` is gitignored (never committed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.research.mcs_seq import mcs_report, write_mcs_receipt
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.vol_bench import (
    VolShard,
    _eval_shard_model,
    build_origins,
    qlike_loss,
    resolve_vol_models,
)

BARS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/file_us_wide/bronze/bars.parquet")
SYMBOL = "NVDA"
HORIZON = 1
MIN_HISTORY = 500
N_ORIGINS = 200
STRIDE = 5
ALPHA = 0.05


def load_real_shard(symbol: str, bars: Path) -> VolShard:
    df = (
        pl.read_parquet(bars, columns=["symbol", "event_time", "open", "high", "low", "close"])
        .filter(pl.col("symbol") == symbol)
        .sort("event_time")
    )
    if df.height < MIN_HISTORY + 100:
        raise ValueError(f"{symbol} tape too short: {df.height}")
    close = df["close"].to_numpy().astype(float)
    high = df["high"].to_numpy().astype(float)
    low = df["low"].to_numpy().astype(float)
    rets = np.diff(np.log(close))
    rv = rets**2  # daily RV proxy on EOD tape
    parkinson = np.log(high[1:] / low[1:]) ** 2 / (4.0 * np.log(2.0))
    if not (np.isfinite(rets).all() and np.isfinite(rv).all() and np.isfinite(parkinson).all()):
        raise ValueError("non-finite derived series on the real tape")
    return VolShard(
        name=f"yahoo_eod:{symbol}",
        returns=rets,
        rv=rv,
        parkinson=parkinson,
        config={
            "data_label": "yahoo_eod",
            "source": "yahoo",
            "symbol": symbol,
            "n_bars": int(close.size),
            "first": str(df["event_time"][0]),
            "last": str(df["event_time"][-1]),
            "rv_construction": "daily_log_return_squared",
            "parkinson_construction": "ln(H/L)^2 / (4 ln 2)",
        },
    )


def main() -> None:
    forecasters = resolve_vol_models(None)
    shard = load_real_shard(SYMBOL, BARS)
    origins = build_origins(
        n_dates=shard.returns.size,
        h=HORIZON,
        min_history=MIN_HISTORY,
        stride=STRIDE,
        n_origins=N_ORIGINS,
    )

    streams: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    for name, forecaster in forecasters.items():
        scored = _eval_shard_model(shard, forecaster, HORIZON, origins, 0)
        if isinstance(scored, str):
            errors[name] = scored
            continue
        targets, forecasts = scored
        streams[name] = qlike_loss(targets, forecasts).tolist()

    receipt = mcs_report(streams, alpha=ALPHA, data_label="yahoo_eod")
    receipt["drill"] = {
        "tape": str(BARS.resolve()),
        "shard": shard.config,
        "horizon": HORIZON,
        "min_history": MIN_HISTORY,
        "n_origins": int(origins.size),
        "stride": STRIDE,
        "model_errors": errors,
        "source": "vol_bench._eval_shard_model per-origin QLIKE on real tape",
    }
    path = write_mcs_receipt(receipt, Path("receipts"))
    final = Path("receipts/mcs_real_drill.json")
    path.rename(final)
    ok = verify_receipt_file(final)
    print(
        json.dumps(
            {
                "receipt": str(final),
                "verify": ok["valid"],
                "data_label": receipt["data_label"],
                "survivors": receipt["survivors"],
                "eliminated": receipt["eliminated"],
                "champion": receipt["champion"],
                "model_errors": errors,
            },
            indent=2,
        )
    )
    if not ok["valid"]:
        raise SystemExit(f"sealed receipt failed verification: {ok['errors']}")


if __name__ == "__main__":
    main()
