"""drift_alarm drill on REAL tape: anytime-valid level-shift on real streams.

Cross-lane corroboration: the calibration audit alarmed on gaussian's
PITs at origin ~17 and changepoint localized [11,46] — ``monitor_stream``
runs the e-process + Page–Hinkley pair on the same PIT stream and on the
``pinball(gaussian) − pinball(conf_t)`` winner-gap stream (changepoint
τ̂=220). Seals one ``drift_alarm.v1`` receipt per stream under
``receipts/drift_real_drill_<stream>.json`` stamped
``data_label=yahoo_eod``. Proper scores only; no P&L claims.

Usage: ``python -m scripts.drift_real_drill [bars.parquet]`` — ``data/``
is gitignored; pass an absolute path where the tape lives outside the
checkout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.metrics.scoring import pinball_loss, pit_values
from quant_fund.research.drift_alarm import monitor_stream
from quant_fund.research.fleet_eval import DEFAULT_TAUS, SyntheticShard, fleet_head_factories
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

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
TAUS = np.asarray(DEFAULT_TAUS, dtype=float)


def real_shard(symbol: str, bars: Path) -> SyntheticShard:
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
    targets = rets[1:]
    feats = rets[:-1].reshape(-1, 1)  # x_t = y_{t-1}: strictly causal
    return SyntheticShard(
        name=f"yahoo_eod:{symbol}",
        x=feats,
        y=targets,
        config={
            "data_label": "yahoo_eod",
            "source": "yahoo",
            "symbol": symbol,
            "n_bars": int(df.height),
            "first": str(df["event_time"][0]),
            "last": str(df["event_time"][-1]),
        },
    )


def _head_panel(shard: SyntheticShard, name: str) -> np.ndarray:
    model = fleet_head_factories(DEFAULT_TAUS, 0)[name]()
    model.fit(shard.x[:N_TRAIN], shard.y[:N_TRAIN])
    if getattr(model, "fleet_lagged_predict", False):
        lag_x = shard.y[N_TRAIN - 1 : N_TRAIN + N_EVAL - 1].reshape(-1, 1)
        return np.asarray(model.predict(lag_x), dtype=float)
    return np.asarray(model.predict(shard.x[N_TRAIN : N_TRAIN + N_EVAL]), dtype=float)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _seal(report: dict, out: Path):
    report.pop("code_revision", None)
    report.pop("meta", None)
    canonical = json.loads(canonical_json_bytes(dict(report)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return verify_receipt_file(out)


def main() -> None:
    shard = real_shard(SYMBOL, BARS)
    y_eval = np.asarray(shard.y[N_TRAIN : N_TRAIN + N_EVAL], dtype=float)
    q_g = _head_panel(shard, "gaussian")
    q_c = _head_panel(shard, "conf_t")

    streams: dict[str, np.ndarray] = {}
    u = np.asarray(pit_values(y_eval, q_g, TAUS), dtype=float)
    streams["gaussian_pit"] = np.clip(u, 1e-9, 1.0 - 1e-9)
    lg = np.stack(
        [pinball_loss(y_eval, q_g[:, i], float(TAUS[i])) for i in range(TAUS.size)], axis=1
    ).mean(axis=1)
    lc = np.stack(
        [pinball_loss(y_eval, q_c[:, i], float(TAUS[i])) for i in range(TAUS.size)], axis=1
    ).mean(axis=1)
    streams["gaussian_minus_conf_t_pinball"] = lg - lc

    results = {}
    for name, stream in streams.items():
        report = monitor_stream(stream.tolist(), data_label="yahoo_eod")
        report["stream"] = name
        report["drill"] = {
            "tape": str(BARS),
            "shard": shard.config,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "feature_frame": "x_t = y_{t-1} (causal lag, fleet_lagged_predict convention)",
        }
        out = OUT_DIR / f"drift_real_drill_{name}.json"
        ok = _seal(report, out)
        results[name] = {
            "receipt": str(out),
            "verify": ok["valid"],
            "eprocess": report["eprocess"],
            "page_hinkley": report["page_hinkley"],
        }
        if not ok["valid"]:
            raise SystemExit(f"sealed receipt failed verification: {ok['errors']}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
