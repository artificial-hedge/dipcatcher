"""Wave-605 motivic-8 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w605 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_steenrod": b.bench_motivic_steenrod_family(),
        "motivic_adem": b.bench_motivic_adem_family(),
        "power_operations": b.bench_power_operations_family(),
        "simplicial_motive": b.bench_simplicial_motive_family(),
        "dk_motive": b.bench_dk_motive_family(),
        "motivic_transfer": b.bench_motivic_transfer_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
