"""Wave-563 harmonic-maps adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w563 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "harmonic_map": b.bench_harmonic_map_family(),
        "eells_sampson": b.bench_eells_sampson_family(),
        "schoen_uhlenbeck": b.bench_schoen_uhlenbeck_family(),
        "bubbling_hm": b.bench_bubbling_hm_family(),
        "heat_flow_hm": b.bench_heat_flow_hm_family(),
        "sacks_uhlenbeck": b.bench_sacks_uhlenbeck_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
