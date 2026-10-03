"""Wave-422 Lie-theory-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w422 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "weyl_chamber": b.bench_weyl_chamber_family(),
        "root_height": b.bench_root_height_family(),
        "borel_subalgebra": b.bench_borel_subalgebra_family(),
        "levi_factor": b.bench_levi_factor_family(),
        "nilpotent_orbit": b.bench_nilpotent_orbit_family(),
        "verma_module": b.bench_verma_module_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
