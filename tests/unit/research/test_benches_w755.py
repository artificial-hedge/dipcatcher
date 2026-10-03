"""Wave-755 loop-soup adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w755 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "lupu_loop": b.bench_lupu_loop_family(),
        "lejan_loop": b.bench_lejan_loop_family(),
        "kassel_wu": b.bench_kassel_wu_family(),
        "kenyon_wilson": b.bench_kenyon_wilson_family(),
        "barlow_ust": b.bench_barlow_ust_family(),
        "lyons_peres": b.bench_lyons_peres_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
