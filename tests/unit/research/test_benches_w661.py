"""Wave-661 chromatic-6 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w661 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ambidexterity": b.bench_ambidexterity_family(),
        "higher_semiadditivity": b.bench_higher_semiadditivity_family(),
        "tate_height": b.bench_tate_height_family(),
        "dieudonne_module": b.bench_dieudonne_module_family(),
        "honda_formal": b.bench_honda_formal_family(),
        "raynaud_height": b.bench_raynaud_height_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
