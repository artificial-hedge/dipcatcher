"""Wave-309 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w309 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "mceliece_lite": b.bench_mceliece_lite_family(),
        "bike_lite": b.bench_bike_lite_family(),
        "hqc_lite": b.bench_hqc_lite_family(),
        "uov_sig": b.bench_uov_sig_family(),
        "rainbow_sig": b.bench_rainbow_sig_family(),
        "sidh_lite": b.bench_sidh_lite_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
