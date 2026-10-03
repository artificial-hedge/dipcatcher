"""AnytimeMCS drill on REAL tape.

Per-origin mean pinball streams for every distribution head (minus the
network/torch trio excluded across the drill campaign) on the Yahoo NVDA
tape, then ``mcs_report`` → the sequential model confidence set: at every
origin the survivor set contains an optimal head w.p. ≥ 1−α, under
arbitrary dependence and stopping.

Seals ``receipts/mcs_real_drill.json`` (mcs_seq.v1 payload stamped
``data_label=yahoo_eod`` + the drill provenance block) and verify-checks
it. Proper scores only; no P&L claims.

Usage: ``python -m scripts.mcs_real_drill [bars.parquet]``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.metrics.scoring import pinball_loss
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    SyntheticShard,
    fleet_head_factories,
)
from quant_fund.research.mcs_seq import mcs_report
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
EXCLUDED_HEADS = ("nbeats", "nhits", "lgbm_q2")


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


def head_losses(head: str, shard: SyntheticShard) -> np.ndarray:
    model = fleet_head_factories(DEFAULT_TAUS, 0)[head]()
    model.fit(shard.x[:N_TRAIN], shard.y[:N_TRAIN])
    if getattr(model, "fleet_lagged_predict", False):
        lag_x = shard.y[N_TRAIN - 1 : N_TRAIN + N_EVAL - 1].reshape(-1, 1)
        q = np.asarray(model.predict(lag_x), dtype=float)
    else:
        q = np.asarray(model.predict(shard.x[N_TRAIN : N_TRAIN + N_EVAL]), dtype=float)
    y_eval = shard.y[N_TRAIN : N_TRAIN + N_EVAL]
    taus = np.asarray(DEFAULT_TAUS)
    return np.stack(
        [pinball_loss(y_eval, q[:, i], float(taus[i])) for i in range(taus.size)],
        axis=1,
    ).mean(axis=1)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    shard = real_shard(SYMBOL, BARS)
    factories = fleet_head_factories(DEFAULT_TAUS, 0)
    heads = sorted(h for h in factories if h not in EXCLUDED_HEADS)
    streams: dict[str, np.ndarray] = {}
    head_errors: dict[str, str] = {}
    for head in heads:
        try:
            streams[head] = head_losses(head, shard)
        except Exception as exc:  # fail-closed adapters raise without weights
            head_errors[head] = f"{type(exc).__name__}: {exc}"
    receipt = mcs_report(streams, data_label="yahoo_eod")
    receipt["drill"] = {
        "tape": str(BARS),
        "shard": shard.config,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "excluded_heads": list(EXCLUDED_HEADS),
        "head_errors": head_errors,
        "feature_frame": "x_t = y_{t-1} (causal lag, fleet_lagged_predict convention)",
        "score": "mean pinball over DEFAULT_TAUS",
    }
    receipt.pop("code_revision", None)
    receipt.pop("meta", None)
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    out = OUT_DIR / "mcs_real_drill.json"
    _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result = verify_receipt_file(out)
    print(
        json.dumps(
            {
                "receipt": str(out),
                "verify": result["valid"],
                "survivors": receipt["survivors"],
                "eliminated": receipt["eliminated"],
                "champion": receipt["champion"],
            },
            indent=2,
        )
    )
    if not result["valid"]:
        raise SystemExit("sealed receipt failed verification")


if __name__ == "__main__":
    main()
