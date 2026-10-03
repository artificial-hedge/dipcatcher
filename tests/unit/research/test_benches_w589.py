"""Wave-589 topos-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w589 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cartesian_closed": b.bench_cartesian_closed_family(),
        "internal_logic": b.bench_internal_logic_family(),
        "subobject_lattice": b.bench_subobject_lattice_family(),
        "power_object": b.bench_power_object_family(),
        "pretopos": b.bench_pretopos_family(),
        "coherent_topos": b.bench_coherent_topos_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
