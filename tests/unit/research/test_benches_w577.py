"""Wave-577 perverse-sheaves adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w577 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "perverse_sheaf": b.bench_perverse_sheaf_family(),
        "intersection_homology": b.bench_intersection_homology_family(),
        "nearby_cycles": b.bench_nearby_cycles_family(),
        "d_module2": b.bench_d_module2_family(),
        "char_cycle": b.bench_char_cycle_family(),
        "middle_perversity": b.bench_middle_perversity_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
