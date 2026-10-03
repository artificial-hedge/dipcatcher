"""Wave-817 fluctuation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w817 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wiener_hopf_f": b.bench_wiener_hopf_f_family(),
        "ladder_height": b.bench_ladder_height_family(),
        "renewal_measure": b.bench_renewal_measure_family(),
        "overshoot_levy": b.bench_overshoot_levy_family(),
        "levy_fluct": b.bench_levy_fluct_family(),
        "spitzer_levy": b.bench_spitzer_levy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
