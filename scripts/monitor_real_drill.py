"""Real-data monitor drill — anytime-valid monitor lanes over real bars.

Reads a collected bronze/silver bars parquet (e.g. the file_us_wide Yahoo
tape produced by ``collect --source yahoo``), converts each symbol's close
series into daily log returns, wraps each as a ``SyntheticShard`` carrying
its true ``data_label``, and runs ``monitor_fleet`` — the coverage, tail,
calibration, conformal, and drift e-process lanes over real return
streams. Emits a sealed ``monitor_run.v1`` receipt.

The tape itself is not vendored (``data/`` is gitignored); to reproduce,
re-collect the same universe via the data engine and rerun this script.
The receipt pins ``inputs_sha256`` (the row frame), the git revision, and
every lane/head/shard parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY, SyntheticShard
from quant_fund.research.monitor_run import monitor_fleet
from quant_fund.research.receipt_v2 import seal_receipt

_DRILL_HEADS = (
    "empirical",
    "gaussian",
    "skew_t",
    "gmm",
    "isotonic",
    "fhs_skew",
    "conf_t",
    "regime",
)


def _real_shards(
    bars: pl.DataFrame, *, symbols: list[str], min_len: int, data_label: str
) -> dict[str, Any]:
    """One labeled shard per symbol: dummy x + real daily log-return y."""
    closes = (
        bars.filter(pl.col("symbol").is_in(symbols))
        .sort("event_time")
        .select(["symbol", "event_time", "close"])
    )
    shards: dict[str, Any] = {}
    for sym in symbols:
        px = closes.filter(pl.col("symbol") == sym)["close"].to_numpy().astype(float)
        px = px[np.isfinite(px) & (px > 0)]
        if px.size < min_len + 1:
            continue
        y = np.diff(np.log(px))
        cfg = {"data_label": data_label, "source_symbol": sym, "n_obs": int(y.size)}
        shards[f"yahoo_{sym.lower()}"] = lambda n, s, y=y, cfg=cfg, sym=sym: SyntheticShard(
            name=f"yahoo_{sym.lower()}",
            x=np.ones((y.size, 1)),
            y=y,
            config=dict(cfg),
        )
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-symbols", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bars = pl.read_parquet(args.bars)
    top = (
        bars.group_by("symbol")
        .agg(pl.len())
        .filter(pl.col("len") >= args.n_train + args.n_eval + 1)
        .sort(["len", "symbol"], descending=[True, False])
        .head(args.n_symbols)["symbol"]
        .to_list()
    )
    if not top:
        raise SystemExit("no symbol meets the length floor")
    label = str(bars["source"].unique().to_list()[0] or "UNKNOWN") + "_eod"
    shards = _real_shards(
        bars,
        symbols=top,
        min_len=args.n_train + args.n_eval,
        data_label=label,
    )
    factories = {
        name: (
            lambda name=name: FLEET_HEAD_REGISTRY[name](
                [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], args.seed
            )
        )
        for name in _DRILL_HEADS
    }
    frame, receipt = monitor_fleet(
        factories,
        shards,
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
    )
    receipt.pop("code_revision", None)
    receipt.pop("meta", None)
    sealed = seal_receipt(receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    alarms = frame.filter(pl.col("status") == "ok")
    n_alarm = sum(alarms[c].sum() for c in alarms.columns if c.endswith("_alarmed"))
    print(
        f"shards={len(shards)} heads={len(factories)} rows={frame.height} "
        f"alarm_flags={int(n_alarm)} label={receipt['data_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
