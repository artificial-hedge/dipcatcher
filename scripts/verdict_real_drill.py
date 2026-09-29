"""Real-data verdict drill — composite honest claim on a real tape.

Same tape handling as ``monitor_real_drill``: each symbol's close series
becomes a labeled ``SyntheticShard`` of daily log returns, then
``run_verdict`` streams every head through the honest-verdict composite —
paired winner's-curse bootstrap, anytime-valid promotion, drift, and tail
agreement — and seals an ``honest_verdict.v1`` receipt whose ``data_label``
is derived from the shards (``yahoo_eod``), never hardcoded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
from quant_fund.research.verdict_run import run_verdict
from scripts.monitor_real_drill import _DRILL_HEADS, _real_shards

from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY
from quant_fund.research.receipt_v2 import seal_receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-symbols", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    bars = pl.read_parquet(args.bars)
    top = (
        bars.group_by("symbol")
        .agg(pl.len())
        .filter(pl.col("len") >= args.n_train + args.n_eval + 1)
        .sort("len", descending=True)
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
    verdict, status = run_verdict(
        factories,
        shards,
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
        n_boot=args.n_boot,
    )
    sealed = seal_receipt(verdict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    print(
        f"shards={len(shards)} heads={len(factories)} verdict={verdict['verdict']} "
        f"label={verdict['data_label']} status_rows={status.height}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
