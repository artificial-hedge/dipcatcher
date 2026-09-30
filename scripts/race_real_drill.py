"""Real-data fleet race — sequential head elimination on a real tape.

Same tape handling as ``monitor_real_drill``: each symbol's close series
becomes a labeled ``SyntheticShard`` of daily log returns, then
``fleet_race`` runs chunked elimination — every head races the chunk-0
incumbent with anytime-valid promote AND demote e-processes. The sealed
``fleet_race.v1`` receipt derives ``data_label`` from the shard configs
(``yahoo_eod``), never hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY, SyntheticShard
from quant_fund.research.fleet_race import fleet_race
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

        def make(n: int, s: int, y: Any = y, cfg: dict = cfg, sym: str = sym) -> SyntheticShard:
            return SyntheticShard(
                name=f"yahoo_{sym.lower()}",
                x=np.ones((y.size, 1)),
                y=y,
                config=dict(cfg),
            )

        shards[f"yahoo_{sym.lower()}"] = make
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-symbols", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--n-chunks", type=int, default=8)
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
    frame, receipt = fleet_race(
        factories,
        shards,
        n_train=args.n_train,
        n_eval=args.n_eval,
        n_chunks=args.n_chunks,
        seed=args.seed,
    )
    for stamp in ("code_revision", "meta", "generated_at", "generated_at_commit", "git_revision"):
        receipt.pop(stamp, None)
    sealed = seal_receipt(receipt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    print(
        f"shards={len(shards)} heads={len(factories)} rows={frame.height} "
        f"label={receipt['data_label']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
