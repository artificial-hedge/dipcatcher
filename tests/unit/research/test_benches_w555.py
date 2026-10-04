"""Wave-555 HMS adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w555 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hms_conjecture": b.bench_hms_conjecture_family(),
        "landau_ginzburg": b.bench_landau_ginzburg_family(),
        "syz_mirror": b.bench_syz_mirror_family(),
        "torus_fibration": b.bench_torus_fibration_family(),
        "wrapped_fukaya": b.bench_wrapped_fukaya_family(),
        "mirror_functor": b.bench_mirror_functor_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
