"""MCS drill: sequential confidence set over vol_bench QLIKE loss streams.

Runs every vol forecaster origin-by-origin on the SYNTHETIC ``garch_vol``
shard (same protocol as ``dipcatcher vol-bench`` — walk-forward, proper
QLIKE losses only) and feeds the per-origin loss streams into
``AnytimeMCS``. Writes a sealed ``receipts/mcs_vol_drill.json``.

Usage: ``python -m scripts.mcs_vol_drill``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_fund.research.mcs_seq import mcs_report, write_mcs_receipt
from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.research.vol_bench import (
    _eval_shard_model,
    build_origins,
    qlike_loss,
    resolve_vol_models,
    resolve_vol_shard_generators,
)

SHARD = "garch_vol"
HORIZON = 1
MIN_HISTORY = 200
N_ORIGINS = 96
STRIDE = 4
N_BARS = MIN_HISTORY + N_ORIGINS * STRIDE + HORIZON + 8
SEED = 7
ALPHA = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("receipts"))
    out_dir = ap.parse_args().out_dir

    forecasters = resolve_vol_models(None)
    shard = resolve_vol_shard_generators([SHARD])[SHARD](N_BARS, SEED)
    origins = build_origins(
        n_dates=N_BARS, h=HORIZON, min_history=MIN_HISTORY, stride=STRIDE, n_origins=N_ORIGINS
    )
    assert origins.size == N_ORIGINS

    streams: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    for name, forecaster in forecasters.items():
        scored = _eval_shard_model(shard, forecaster, HORIZON, origins, SEED)
        if isinstance(scored, str):
            errors[name] = scored
            continue
        targets, forecasts = scored
        streams[name] = qlike_loss(targets, forecasts).tolist()

    receipt = mcs_report(streams, alpha=ALPHA, data_label="SYNTHETIC")
    receipt["drill"] = {
        "shard": SHARD,
        "horizon": HORIZON,
        "min_history": MIN_HISTORY,
        "n_origins": int(origins.size),
        "stride": STRIDE,
        "seed": SEED,
        "model_errors": errors,
        "source": "vol_bench._eval_shard_model per-origin QLIKE",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_mcs_receipt(receipt, out_dir)
    final = out_dir / "mcs_vol_drill.json"
    path.rename(final)
    ok = verify_receipt_file(final)
    print(
        json.dumps(
            {
                "receipt": str(final),
                "verify": ok["valid"],
                "survivors": receipt["survivors"],
                "eliminated": receipt["eliminated"],
                "champion": receipt["champion"],
                "model_errors": errors,
            },
            indent=2,
        )
    )
    if not ok["valid"]:
        raise SystemExit(f"sealed receipt failed verification: {ok['errors']}")


if __name__ == "__main__":
    main()
