"""Wave-641 homotopy-17 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w641 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "smash_prod": b.bench_smash_prod_family(),
        "stable_stem2": b.bench_stable_stem2_family(),
        "homotopy_colim": b.bench_homotopy_colim_family(),
        "unstable_tower": b.bench_unstable_tower_family(),
        "periodic_fam": b.bench_periodic_fam_family(),
        "chromatic_htpy": b.bench_chromatic_htpy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
