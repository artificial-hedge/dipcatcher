"""Wave-813 diffusion adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w813 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "feller_boundary": b.bench_feller_boundary_family(),
        "scale_measure": b.bench_scale_measure_family(),
        "speed_measure": b.bench_speed_measure_family(),
        "diffusion_semigroup": b.bench_diffusion_semigroup_family(),
        "yosida_op": b.bench_yosida_op_family(),
        "kreyn_resolvent": b.bench_kreyn_resolvent_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
