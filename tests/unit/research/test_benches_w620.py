"""Wave-620 stacks-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w620 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gerbe2": b.bench_gerbe2_family(),
        "band_gerbe": b.bench_band_gerbe_family(),
        "rigid_stack": b.bench_rigid_stack_family(),
        "dm_stack2": b.bench_dm_stack2_family(),
        "inertia_stack": b.bench_inertia_stack_family(),
        "root_stack": b.bench_root_stack_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
