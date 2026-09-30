"""conformal_monitor drill on REAL tape.

Runs Vovk conformal martingales over two real-tape streams derived from the
Yahoo NVDA bars (same construction as the other real drills):

- ``gaussian_pit``: the gaussian head's PIT values — a correctly calibrated
  head yields exchangeable PITs, so an alarm here says the head's
  distribution is wrong (not merely mis-scaled).
- ``gaussian_minus_conf_t_pinball``: per-origin pinball-loss difference —
  a shift here means the head-vs-head gap regime changed mid-stream.

Each stream seals a ``conformal_monitor.v1`` receipt under
``receipts/conformal_real_drill_<stream>.json`` stamped
``data_label=yahoo_eod`` and immediately ``verify_receipt_file``-checked.
Proper scores only; no P&L claims.

Usage: ``python -m scripts.conformal_real_drill [bars.parquet]`` —
``data/`` is gitignored; pass an absolute path where the tape lives
outside the checkout.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.metrics.scoring import pinball_loss, pit_values
from quant_fund.research.conformal_monitor import monitor
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    SyntheticShard,
    fleet_head_factories,
)
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


def head_quantiles(head: str, shard: SyntheticShard) -> np.ndarray:
    model = fleet_head_factories(DEFAULT_TAUS, 0)[head]()
    model.fit(shard.x[:N_TRAIN], shard.y[:N_TRAIN])
    if getattr(model, "fleet_lagged_predict", False):
        # lag-1 observed return is the predict matrix — strictly causal
        lag_x = shard.y[N_TRAIN - 1 : N_TRAIN + N_EVAL - 1].reshape(-1, 1)
        return np.asarray(model.predict(lag_x), dtype=float)
    return np.asarray(model.predict(shard.x[N_TRAIN : N_TRAIN + N_EVAL]), dtype=float)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _seal(report: dict[str, object], name: str):
    report.pop("code_revision", None)
    report.pop("meta", None)
    canonical = json.loads(canonical_json_bytes(dict(report)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    out = OUT_DIR / f"conformal_real_drill_{name}.json"
    _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return verify_receipt_file(out), out


def main() -> None:
    shard = real_shard(SYMBOL, BARS)
    y_eval = shard.y[N_TRAIN : N_TRAIN + N_EVAL]
    q_gauss = head_quantiles("gaussian", shard)
    q_conft = head_quantiles("conf_t", shard)
    taus = np.asarray(DEFAULT_TAUS)

    pits = np.clip(pit_values(y_eval, q_gauss, taus), 1e-6, 1.0 - 1e-6)

    def _mean_pinball(q: np.ndarray) -> np.ndarray:
        return np.stack(
            [pinball_loss(y_eval, q[:, i], float(taus[i])) for i in range(taus.size)],
            axis=1,
        ).mean(axis=1)

    diff = _mean_pinball(q_gauss) - _mean_pinball(q_conft)

    results = {}
    for name, stream in (("gaussian_pit", pits), ("gaussian_minus_conf_t_pinball", diff)):
        report = monitor(stream.tolist(), data_label="yahoo_eod")
        report["drill"] = {
            "stream": name,
            "tape": str(BARS),
            "shard": shard.config,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "feature_frame": "x_t = y_{t-1} (causal lag, fleet_lagged_predict convention)",
        }
        ok, out = _seal(report, name)
        results[name] = {
            "receipt": str(out),
            "verify": ok["valid"],
            "alarmed": report["alarmed"],
            "alarm_index": report["alarm_index"],
            "final_martingale": report["final_martingale"],
        }
    print(json.dumps(results, indent=2))
    bad = [k for k, v in results.items() if not v["verify"]]
    if bad:
        raise SystemExit(f"sealed receipts failed verification: {bad}")


if __name__ == "__main__":
    main()
