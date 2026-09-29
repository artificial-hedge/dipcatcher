"""emerge drill on REAL tape: pool the real-tape e-values into one claim.

The sealed drill receipts store only ``inputs_sha256`` of their row
frames, so this script re-runs the same audits (same tape, same params)
and pools each head's per-lane final e-values with the mergers that stay
valid under ARBITRARY DEPENDENCE (``emerge_mean``, ``emerge_harmonic``) —
the lanes share one tape, so the product rule's independence assumption
does NOT hold; it is reported flagged as diagnostics-only.
``emerge_bonferroni`` over heads gives the family-level e-value. Seals
``receipts/emerge_real_drill.json`` stamped ``data_label=yahoo_eod``.
Proper scores only; no P&L claims.

Usage: ``python -m scripts.emerge_real_drill [bars.parquet]`` —
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

from quant_fund.research.coverage_watch import audit_interval_coverage
from quant_fund.research.emerge import (
    emerge_bonferroni,
    emerge_harmonic,
    emerge_mean,
    emerge_product,
)
from quant_fund.research.fleet_eval import DEFAULT_TAUS, SyntheticShard, fleet_head_factories
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.tail_watch import audit_tail_depth
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

BARS = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("data/file_us_wide/bronze/bars.parquet")
)
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
    gen = {shard.name: lambda n, seed, s=shard: s}
    cov_f, _ = audit_interval_coverage(
        factories, gen, n_train=N_TRAIN, n_eval=N_EVAL, data_label="yahoo_eod"
    )
    tail_f, _ = audit_tail_depth(
        factories, gen, n_train=N_TRAIN, n_eval=N_EVAL, data_label="yahoo_eod"
    )

    per_head: dict[str, dict[str, float]] = {}
    for row in cov_f.filter(pl.col("status") == "ok").iter_rows(named=True):
        lane = f"coverage@{row['level']}"
        per_head.setdefault(row["head"], {})[lane] = float(row["final_evalue"])
    for row in tail_f.filter(pl.col("status") == "ok").iter_rows(named=True):
        per_head.setdefault(row["head"], {})["tail_depth"] = float(row["final_evalue"])

    pooled = {}
    for head, lanes in per_head.items():
        ev = list(lanes.values())
        pooled[head] = {
            "lanes": lanes,
            "e_mean": emerge_mean(ev),
            "e_harmonic": emerge_harmonic(ev),
            "e_product_diagnostics_only": emerge_product(ev),
            "note": "product assumes independence — INVALID here (shared tape); "
            "mean/harmonic are the valid pooled claims",
        }
    family_e = emerge_bonferroni([v["e_mean"] for v in pooled.values()])

    report = {
        "kind": "emerge_drill.v1",
        "schema": "emerge_drill.v1",
        "data_label": "yahoo_eod",
        "research_only": True,
        "live_pnl_claim": False,
        "n_heads": len(pooled),
        "pooled_per_head": pooled,
        "family_evalue": family_e,
        "drill": {
            "tape": str(BARS.resolve()),
            "shard": shard.config,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "excluded_heads": list(EXCLUDED_HEADS),
            "feature_frame": "x_t = y_{t-1} (causal lag, fleet_lagged_predict convention)",
        },
        "claim": (
            "under arbitrary dependence across lanes (shared tape), the pooled "
            "evidence against head h's interval honesty is e_mean/e_harmonic; "
            "family-level e-value via e-Bonferroni"
        ),
        "evidence": [
            "e_value_merge_under_arbitrary_dependence",
            "harmonic_mean_valid",
            "product_rule_flagged_dependence_violation",
            "proper_scores_only",
        ],
    }
    canonical = json.loads(canonical_json_bytes(dict(report)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    out = Path("receipts") / "emerge_real_drill.json"
    _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ok = verify_receipt_file(out)
    print(
        json.dumps(
            {
                "receipt": str(out),
                "verify": ok["valid"],
                "family_evalue": family_e,
                "n_heads": len(pooled),
            },
            indent=2,
        )
    )
    for h, v in sorted(pooled.items(), key=lambda kv: -kv[1]["e_mean"])[:6]:
        print(f"{h}: e_mean={v['e_mean']:.3g} lanes={list(v['lanes'])}")
    if not ok["valid"]:
        raise SystemExit(f"sealed receipt failed verification: {ok['errors']}")


if __name__ == "__main__":
    main()
