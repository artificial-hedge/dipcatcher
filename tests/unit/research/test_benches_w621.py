"""Wave-621 motivic-10 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w621 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_k": b.bench_motivic_k_family(),
        "motivic_borel": b.bench_motivic_borel_family(),
        "motivic_height": b.bench_motivic_height_family(),
        "motivic_chow": b.bench_motivic_chow_family(),
        "motivic_homology": b.bench_motivic_homology_family(),
        "motivic_class": b.bench_motivic_class_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
