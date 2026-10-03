"""Wave-482 chromatic-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w482 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "chromatic_fracture": b.bench_chromatic_fracture_family(),
        "morava_stabilizer": b.bench_morava_stabilizer_family(),
        "fgsl_group": b.bench_fgsl_group_family(),
        "tate_spec": b.bench_tate_spec_family(),
        "blue_shift": b.bench_blue_shift_family(),
        "red_shift": b.bench_red_shift_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
