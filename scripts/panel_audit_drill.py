"""Cross-sectional panel audit on the REAL tape.

Every other real drill watches one symbol (NVDA). This one watches a
stride-sampled cross-section of the Yahoo universe: for each sampled
symbol the 80% central interval breach stream of every head goes through
its own ``CoverageEProcess`` (coverage_watch lane), giving a per-cell
anytime-valid e-value.

Pooling uses the arithmetic mean of e-values — valid under *arbitrary*
cross-sectional dependence (Vovk & Wang 2020 / emerge_mean semantics),
so overlapping trading calendars and factor co-movement cannot inflate it:

- per-head pooled e over symbols: "this head is miscalibrated across the
  tape" — a family claim per head.
- per-symbol pooled e over heads: "this name's tape is anomalous".
- the family bound uses Bonferroni over heads: ``n_heads * max_h pooled_e``.

Seals ``receipts/panel_audit_real_drill.json`` (kind ``panel_audit.v1``,
``data_label=yahoo_eod``) and verify-checks it. Proper scores only; no
P&L claims.

Usage: ``python -m scripts.panel_audit_drill [bars.parquet]``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.research.coverage_watch import audit_interval_coverage
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    ShardGenerator,
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
N_TRAIN = 1000
N_EVAL = 300
LEVEL = 0.8
N_SYMBOLS = 24
EXCLUDED_HEADS = ("nbeats", "nhits", "lgbm_q2")


def _symbol_universe(bars: Path, n: int) -> list[str]:
    counts = pl.read_parquet(bars, columns=["symbol"]).group_by("symbol").len().sort("symbol")
    ok = counts.filter(pl.col("len") >= N_TRAIN + N_EVAL + 5)
    if ok.height < n:
        raise ValueError(f"only {ok.height} symbols have >= {N_TRAIN + N_EVAL + 5} bars")
    # deterministic stride sample over the sorted universe
    stride = ok.height // n
    return [ok["symbol"][i] for i in range(0, ok.height, stride)][:n]


def _symbol_shard(symbol: str, bars: Path) -> SyntheticShard:
    df = (
        pl.read_parquet(bars, columns=["symbol", "event_time", "close"])
        .filter(pl.col("symbol") == symbol)
        .sort("event_time")
    )
    rets = np.diff(np.log(df["close"].to_numpy().astype(float)))
    if not np.isfinite(rets).all():
        raise ValueError(f"non-finite returns on {symbol}")
    targets = rets[1:]
    feats = rets[:-1].reshape(-1, 1)  # x_t = y_{t-1}: strictly causal
    return SyntheticShard(
        name=f"yahoo_eod:{symbol}",
        x=feats,
        y=targets,
        config={"data_label": "yahoo_eod", "source": "yahoo", "symbol": symbol},
    )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> None:
    symbols = _symbol_universe(BARS, N_SYMBOLS)
    factories = {
        h: f for h, f in fleet_head_factories(DEFAULT_TAUS, 0).items() if h not in EXCLUDED_HEADS
    }
    generators: dict[str, ShardGenerator] = {}
    for sym in symbols:

        def _gen(_n: int, _seed: int, _s: str = sym) -> SyntheticShard:
            return _symbol_shard(_s, BARS)

        generators[sym] = _gen
    frame, _ = audit_interval_coverage(
        factories,
        generators,
        levels=(LEVEL,),
        n_train=N_TRAIN,
        n_eval=N_EVAL,
        data_label="yahoo_eod",
    )
    ok = frame.filter(pl.col("status") == "ok")
    if ok.height == 0:
        raise SystemExit("no healthy audit cells")

    # Per-head pooled e-value across symbols (mean is an e-value under
    # arbitrary dependence — emerge_mean semantics).
    head_rows = (
        ok.group_by("head")
        .agg(
            pl.col("final_evalue").mean().alias("pooled_evalue"),
            pl.col("coverage_alarm").mean().alias("alarm_share"),
            pl.col("breach_rate").mean().alias("mean_breach_rate"),
            pl.len().alias("n_symbols"),
        )
        .sort(["pooled_evalue", "head"], descending=[True, False])
    )
    symbol_rows = (
        ok.group_by("shard")
        .agg(
            pl.col("final_evalue").mean().alias("pooled_evalue"),
            pl.col("coverage_alarm").sum().alias("n_heads_alarmed"),
        )
        .sort(["pooled_evalue", "shard"], descending=[True, False])
    )
    heads = sorted(ok["head"].unique().to_list())
    alpha = 0.05
    family_threshold = len(heads) / alpha  # Bonferroni over heads
    receipt: dict[str, object] = {
        "kind": "panel_audit.v1",
        "schema": "panel_audit.v1",
        "data_label": "yahoo_eod",
        "research_only": True,
        "live_pnl_claim": False,
        "level": LEVEL,
        "alpha": alpha,
        "n_symbols": ok["shard"].n_unique(),
        "n_heads": len(heads),
        "n_cells_ok": int(ok.height),
        "n_cells_total": int(frame.height),
        "excluded_heads": list(EXCLUDED_HEADS),
        "params": {
            "data_labels": {"yahoo_eod": int(ok.height)},
            "pooling": "arithmetic_mean_evalues",
        },
        "heads": head_rows.to_dicts(),
        "symbols": symbol_rows.to_dicts(),
        "family": {
            "bonferroni_threshold": family_threshold,
            "max_head_pooled_evalue": float(head_rows["pooled_evalue"][0]),
            "family_alarmed": bool(head_rows["pooled_evalue"][0] >= family_threshold),
        },
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        "drill": {
            "tape": str(BARS),
            "symbols_sampled": symbols,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "feature_frame": "x_t = y_{t-1} (causal lag, fleet_lagged_predict convention)",
        },
        "evidence": [
            "bernoulli_lr_bet",
            "exact_evalue_under_h0",
            "evalue_arithmetic_mean_valid_under_arbitrary_dependence",
            "bonferroni_family_bound",
            "cross_sectional_panel",
        ],
    }
    receipt.pop("code_revision", None)
    receipt.pop("meta", None)
    canonical = json.loads(canonical_json_bytes(dict(receipt)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    out = OUT_DIR / "panel_audit_real_drill.json"
    _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result = verify_receipt_file(out)
    family = receipt["family"]
    assert isinstance(family, dict)
    print(
        json.dumps(
            {
                "receipt": str(out),
                "verify": result["valid"],
                "verify_errors": result.get("errors"),
                "family_alarmed": family["family_alarmed"],
                "top_heads": [
                    {k: r[k] for k in ("head", "pooled_evalue", "alarm_share")}
                    for r in head_rows.to_dicts()[:5]
                ],
            },
            indent=2,
        )
    )
    if not result["valid"]:
        raise SystemExit("sealed receipt failed verification")


if __name__ == "__main__":
    main()
