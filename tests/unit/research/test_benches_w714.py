"""Wave-714 motivic-25 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w714 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_total": b.bench_motivic_total_family(),
        "motivic_partial": b.bench_motivic_partial_family(),
        "motivic_functor": b.bench_motivic_functor_family(),
        "motivic_nerve": b.bench_motivic_nerve_family(),
        "derived_proper2": b.bench_derived_proper2_family(),
        "derived_separated2": b.bench_derived_separated2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
