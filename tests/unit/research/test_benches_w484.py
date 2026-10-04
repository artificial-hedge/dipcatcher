"""Wave-484 synthetic-math-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w484 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "internal_univ": b.bench_internal_univ_family(),
        "virtual_hodge": b.bench_virtual_hodge_family(),
        "stein_space": b.bench_stein_space_family(),
        "formal_model": b.bench_formal_model_family(),
        "univalent_found": b.bench_univalent_found_family(),
        "synth_stable": b.bench_synth_stable_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
