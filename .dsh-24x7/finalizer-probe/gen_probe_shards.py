"""Generate disposable but schema-valid eval shards for the finalizer probe."""
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\dipcatcher\.dsh-24x7\finalizer-probe")
ROOT.mkdir(parents=True, exist_ok=True)

TAUS = [0.05, 0.5, 0.95]
SCORING_CONTRACT = "native_shapes_timesfm_point_first.v2"
TARGETS = ["kronos_small", "chronos2", "bolt_small", "timesfm"]
MODELS = ["dip_garch_t", "dip_fhs"] + TARGETS

GROUPS = {
    "d1": dict(
        assets=["adausdt_1d", "avaxusdt_1d", "bnbusdt_1d_deep", "btcusdt_1d_deep",
                "dogeusdt_1d", "ethusdt_1d_deep", "linkusdt_1d", "ltcusdt_1d",
                "solusdt_1d_deep", "trxusdt_1d", "xrpusdt_1d_deep"],
        seed=7, interval_ns=86_400_000_000_000,
    ),
    "h4": dict(
        assets=["bnbusdt_4h", "btcusdt_4h_deep", "ethusdt_4h_deep",
                "solusdt_4h_deep", "xrpusdt_4h"],
        seed=7, interval_ns=14_400_000_000_000,
    ),
    "seed11": dict(
        assets=["btcusdt_1d_deep", "ethusdt_1d_deep", "solusdt_1d_deep"],
        seed=11, interval_ns=86_400_000_000_000,
    ),
    "seed23": dict(
        assets=["btcusdt_1d_deep", "ethusdt_1d_deep", "solusdt_1d_deep"],
        seed=23, interval_ns=86_400_000_000_000,
    ),
}

N_ROWS = 30
BASE_NS = 1_700_000_000_000_000_000

for prefix, spec in GROUPS.items():
    for ai, asset in enumerate(spec["assets"]):
        rng = np.random.default_rng(abs(hash((prefix, asset))) % (2**32))
        n = N_ROWS
        m = len(MODELS)
        crps = rng.uniform(0.05, 1.5, size=(n, m)).astype(np.float64)
        pinball = rng.uniform(0.01, 0.9, size=(n, m, len(TAUS))).astype(np.float64)
        asset_ids = np.zeros(n, dtype=np.int64)
        times = BASE_NS + spec["interval_ns"] * np.arange(n, dtype=np.int64)
        fake_bars = {asset: hashlib.sha256(f"bars:{asset}".encode()).hexdigest()}
        fake_art = {t: hashlib.sha256(f"artifact:{t}".encode()).hexdigest()
                    for t in TARGETS}
        meta = {
            "asset_names": [asset],
            "targets": TARGETS,
            "config": {
                "lookback": 120,
                "window": 300,
                "garch_window": 500,
                "samples_per_origin": 20,
                "origins_per_asset": n,
                "seed": spec["seed"],
                "taus": TAUS,
            },
            "bars_sha256": fake_bars,
            "artifact_sha256": fake_art,
            "bar_interval_ns": spec["interval_ns"],
            "scoring_contract": SCORING_CONTRACT,
        }
        out = ROOT / f"{prefix}_{asset}.losses.npz"
        np.savez(
            out,
            crps_matrix=crps,
            pinball_cube=pinball,
            asset_ids=asset_ids,
            model_names=np.asarray(MODELS),
            target_time_ns=times,
            meta_json=np.asarray(json.dumps(meta)),
        )
        print("wrote", out.name)

print("done")
