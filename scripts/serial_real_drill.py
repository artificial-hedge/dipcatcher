"""serial_watch drill on REAL tape: PIT serial-independence per fleet head.

Runs each lightweight distribution head on one causal eval window of the
collected Yahoo EOD tape, converts realized-vs-quantile-panel pairs to PITs
via ``pit_values`` (linear interpolation of the empirical CDF), and audits
each head's PIT stream for lag-1..5 serial dependence with ``serial_report``.
One sealed ``serial_watch_<digest>.json`` per head, stamped
``data_label=yahoo_eod``. Proper scores only; no P&L claims.

Usage: ``python -m scripts.serial_real_drill [bars.parquet]`` — ``data/`` is
gitignored; pass an absolute path on machines where the tape lives outside
the checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.metrics.scoring import pit_values
from quant_fund.research.fleet_eval import DEFAULT_TAUS, fleet_head_factories
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.serial_watch import serial_report, write_serial_receipt

BARS = Path("data/file_us_wide/bronze/bars.parquet")
OUT_DIR = Path("receipts")
_args = sys.argv[1:]
if "--out" in _args:
    _i = _args.index("--out")
    OUT_DIR = Path(_args[_i + 1])
    _args = _args[:_i] + _args[_i + 2 :]
_pos = [a for a in _args if not a.startswith("--")]
if _pos:
    BARS = Path(_pos[0])
SYMBOL = "NVDA"
N_TRAIN = 1000
N_EVAL = 300
# Torch-backed heads (nbeats/nhits) and the lgbm ensemble are excluded from
# this drill — heavyweight for a serial audit; the exclusion is recorded in
# the summary, not silently dropped.
EXCLUDED_HEADS = ("nbeats", "nhits", "lgbm_q2")
PIT_EPS = 1e-9


def load_real_series(symbol: str, bars: Path) -> tuple[np.ndarray, dict[str, Any]]:
    df = (
        pl.read_parquet(bars, columns=["symbol", "event_time", "close"])
        .filter(pl.col("symbol") == symbol)
        .sort("event_time")
    )
    if df.height < N_TRAIN + N_EVAL + 5:
        raise ValueError(f"{symbol} tape too short: {df.height}")
    rets = np.diff(np.log(df["close"].to_numpy().astype(float)))
    if not np.isfinite(rets).all():
        raise ValueError("non-finite returns on the real tape")
    meta = {
        "data_label": "yahoo_eod",
        "source": "yahoo",
        "symbol": symbol,
        "n_bars": int(df.height),
        "first": str(df["event_time"][0]),
        "last": str(df["event_time"][-1]),
    }
    return rets, meta


def eval_head(factory: Any, y: np.ndarray) -> np.ndarray:
    """Fit once on the first N_TRAIN returns, then predict the eval window.

    Mirrors ``fleet_eval._score_row``: heads with ``fleet_lagged_predict``
    consume the observed lag-1 return as their predict matrix; others take a
    causal feature frame built as ``x_t = y_{t-1}``.
    """
    model = factory()
    targets = y[1:]
    feats = y[:-1].reshape(-1, 1)  # x_t = y_{t-1}: strictly causal
    x_tr, y_tr = feats[:N_TRAIN], targets[:N_TRAIN]
    model.fit(x_tr, y_tr)
    # Eval window: for target y_t the causal feature is y_{t-1}, which is
    # also what fleet_lagged_predict heads consume — one slice serves both.
    x_ev = feats[N_TRAIN : N_TRAIN + N_EVAL]
    q = np.asarray(model.predict(x_ev), dtype=float)
    if q.ndim != 2 or q.shape[1] != len(DEFAULT_TAUS):
        raise ValueError(f"predict returned shape {q.shape}")
    if not np.isfinite(q).all() or np.any(np.diff(q, axis=1) < 0.0):
        raise ValueError("non-finite or crossing quantiles")
    y_ev = targets[N_TRAIN : N_TRAIN + N_EVAL]
    return pit_values(y_ev, q, np.asarray(DEFAULT_TAUS))


def main() -> None:
    y, meta = load_real_series(SYMBOL, BARS)
    factories = {
        k: v for k, v in fleet_head_factories(DEFAULT_TAUS, 0).items() if k not in EXCLUDED_HEADS
    }
    summary: dict[str, Any] = {
        "meta": meta,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "excluded_heads": list(EXCLUDED_HEADS),
        "heads": {},
    }
    for name, factory in factories.items():
        try:
            u = eval_head(factory, y)
        except Exception as exc:  # noqa: BLE001 — per-head errors are evidence
            summary["heads"][name] = {"error": str(exc)}
            print(f"{name}: ERROR {exc}")
            continue
        n_nan = int(np.isnan(u).sum())
        u_clean = np.clip(u[np.isfinite(u)], PIT_EPS, 1.0 - PIT_EPS)
        receipt = serial_report(u_clean.tolist(), data_label="yahoo_eod")
        receipt["drill"] = {
            "head": name,
            "shard": meta,
            "n_train": N_TRAIN,
            "n_pit": int(u_clean.size),
            "n_pit_nan_dropped": n_nan,
            "pit_construction": "pit_values interp, clipped to (1e-9, 1-1e-9)",
        }
        receipt.pop("code_revision", None)
        receipt.pop("meta", None)
        path = write_serial_receipt(receipt, OUT_DIR)
        ok = verify_receipt_file(path)
        summary["heads"][name] = {
            "receipt": path.name,
            "verify": ok["valid"],
            "alarmed_lags": receipt["alarmed_lags"],
            "pooled_evalue": receipt["pooled_evalue"],
            "pooled_alarmed": receipt["pooled_alarmed"],
        }
        print(
            f"{name}: lags={receipt['alarmed_lags']} pooled_e={receipt['pooled_evalue']:.3g} "
            f"verify={ok['valid']}"
        )
        if not ok["valid"]:
            raise SystemExit(f"{name} receipt failed verification: {ok['errors']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
