"""Wave-603 topos-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w603 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "slice_topos": b.bench_slice_topos_family(),
        "logical_morph": b.bench_logical_morph_family(),
        "classifying_topos": b.bench_classifying_topos_family(),
        "atomic_topos": b.bench_atomic_topos_family(),
        "essential_morph": b.bench_essential_morph_family(),
        "giraud_axiom": b.bench_giraud_axiom_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
