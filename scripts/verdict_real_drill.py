"""honest_verdict drill on REAL tape: composite claim over fleet heads.

Runs every lightweight fleet head on the collected Yahoo EOD tape
(x_t = y_{t-1}, the causal frame ``fleet_eval`` gives
``fleet_lagged_predict`` heads), builds the per-origin mean-pinball loss
stream + PIT stream per head, and hands both to ``honest_verdict`` — the
capstone that composites winner's-curse correction, anytime-valid
promotion, drift, magnitude CS, and PIT calibration into ONE sealed claim.
Writes ``receipts/verdict_real_drill.json`` stamped
``data_label=yahoo_eod``. Proper scores only; no P&L claims.

Usage: ``python -m scripts.verdict_real_drill [bars.parquet]`` —
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
from quant_fund.research.fleet_eval import (
    DEFAULT_TAUS,
    SyntheticShard,
    fleet_head_factories,
)
from quant_fund.research.honest_verdict import honest_verdict
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
    taus = np.asarray(DEFAULT_TAUS, dtype=float)
    y_eval = np.asarray(shard.y[N_TRAIN : N_TRAIN + N_EVAL], dtype=float)
    scores: dict[str, np.ndarray] = {}
    pits: dict[str, np.ndarray] = {}
    head_errors: dict[str, str] = {}
    for name in sorted(fleet_head_factories(DEFAULT_TAUS, 0)):
        if name in EXCLUDED_HEADS:
            continue
        try:
            model = fleet_head_factories(DEFAULT_TAUS, 0)[name]()
            model.fit(shard.x[:N_TRAIN], shard.y[:N_TRAIN])
            if getattr(model, "fleet_lagged_predict", False):
                lag_x = shard.y[N_TRAIN - 1 : N_TRAIN + N_EVAL - 1].reshape(-1, 1)
                q = np.asarray(model.predict(lag_x), dtype=float)
            else:
                q = np.asarray(model.predict(shard.x[N_TRAIN : N_TRAIN + N_EVAL]), dtype=float)
            losses = np.stack(
                [pinball_loss(y_eval, q[:, i], float(taus[i])) for i in range(taus.size)],
                axis=1,
            )
            finite = np.isfinite(losses).all(axis=1) & np.isfinite(q).all(axis=1)
            scores[name] = np.asarray(losses[finite].mean(axis=1), dtype=float)
            u = np.asarray(pit_values(y_eval[finite], q[finite], taus), dtype=float)
            pits[name] = np.clip(u, 1e-9, 1.0 - 1e-9)
        except Exception as exc:
            head_errors[name] = str(exc)
    n_min = min(len(v) for v in scores.values()) if scores else 0
    scores = {h: v[:n_min] for h, v in scores.items()}
    pits = {h: v[:n_min] for h, v in pits.items()}

    report = honest_verdict(scores, pits=pits, data_label="yahoo_eod")
    # verifier contract: declared per-source labels let a real label prove
    # itself — without them the kind falls back to SYNTHETIC-only.
    report["run"] = {
        "params": {"data_labels": {h: "yahoo_eod" for h in scores}},
    }
    report["drill"] = {
        "tape": str(BARS),
        "shard": shard.config,
        "n_train": N_TRAIN,
        "n_eval": N_EVAL,
        "n_stream": n_min,
        "excluded_heads": list(EXCLUDED_HEADS),
        "head_errors": head_errors,
        "loss_stream": "per-origin mean pinball over the default tau grid",
        "computed_on": "seq-union scratch (all verdict lanes present); isolated lane branches degrade to inconclusive",
        "feature_frame": "x_t = y_{t-1} (causal lag, fleet_lagged_predict convention)",
    }
    report.pop("code_revision", None)
    report.pop("meta", None)
    canonical = json.loads(canonical_json_bytes(dict(report)))
    digest = hash_bytes(canonical_json_bytes(canonical))
    payload = {**canonical, "receipt_sha256": digest}
    out = OUT_DIR / "verdict_real_drill.json"
    _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ok = verify_receipt_file(out)

    print(
        json.dumps(
            {
                "receipt": str(out),
                "verify": ok["valid"],
                "verdict": report["verdict"],
                "winner": report["winner"],
                "n_heads": report["n_heads"],
                "n_obs": report["n_obs"],
                "unavailable_lanes": report["unavailable_lanes"],
                "head_errors": head_errors,
                "mean_pinball": {h: round(float(v.mean()), 6) for h, v in sorted(scores.items())},
            },
            indent=2,
        )
    )
    if not ok["valid"]:
        raise SystemExit(f"sealed receipt failed verification: {ok['errors']}")


if __name__ == "__main__":
    main()
