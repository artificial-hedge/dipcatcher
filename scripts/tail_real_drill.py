"""tail_audit drill on REAL tape: nested-quantile tail-depth per head.

Complements ``coverage_real_drill``: coverage_watch asks whether heads
breach at the nominal rate; tail_watch asks whether the breaches that
occur land as deep as the reported distribution claims — among
q_0.10-breaches, exactly half must also breach q_0.05 under ANY correct
conditional tail. Feeds the collected Yahoo EOD tape to
``audit_tail_depth`` as a custom shard (x_t = y_{t-1}, the causal frame
``fleet_eval`` gives ``fleet_lagged_predict`` heads). Writes a sealed
``receipts/tail_real_drill.json`` stamped ``data_label=yahoo_eod``.
Proper scores only; no P&L claims.

Usage: ``python -m scripts.tail_real_drill [bars.parquet]`` — ``data/``
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

from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    SyntheticShard,
    fleet_head_factories,
)
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.tail_watch import audit_tail_depth
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
EXCLUDED_HEADS = ("nbeats", "nhits", "lgbm_q2")  # heavyweight; exclusion is reported


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


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    shard = real_shard(SYMBOL, BARS)
    factories = {
        k: v for k, v in fleet_head_factories(DEFAULT_TAUS, 0).items() if k not in EXCLUDED_HEADS
    }
    frame, receipt = audit_tail_depth(
        factories,
        {shard.name: lambda n, seed, s=shard: s},
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        data_label="yahoo_eod",
    )
    receipt["drill"] = {
        "tape": str(BARS),
        "shard": shard.config,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "excluded_heads": list(EXCLUDED_HEADS),
        "feature_frame": "x_t = y_{t-1} (causal lag, fleet_lagged_predict convention)",
    }
    receipt.pop("code_revision", None)
    receipt.pop("meta", None)
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    out = OUT_DIR / "tail_real_drill.json"
    _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ok = verify_receipt_file(out)

    alarms = frame.filter(pl.col("tail_alarm")).select(["head", "deep_share", "final_evalue"])
    print(
        frame.select(
            ["head", "status", "n_outer", "n_deep", "deep_share", "final_evalue", "tail_alarm"]
        )
    )
    print(
        json.dumps(
            {"receipt": str(out), "verify": ok["valid"], "alarms": alarms.to_dicts()},
            indent=2,
        )
    )
    if not ok["valid"]:
        raise SystemExit(f"sealed receipt failed verification: {ok['errors']}")


if __name__ == "__main__":
    main()
