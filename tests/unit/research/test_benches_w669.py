"""Wave-669 motivic-16 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w669 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "slice_filtration2": b.bench_slice_filtration2_family(),
        "milnor_operations2": b.bench_milnor_operations2_family(),
        "motivic_bordism": b.bench_motivic_bordism_family(),
        "motivic_eilenberg2": b.bench_motivic_eilenberg2_family(),
        "f_motive2": b.bench_f_motive2_family(),
        "motivic_ss2": b.bench_motivic_ss2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
