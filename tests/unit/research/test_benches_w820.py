"""Wave-820 semimartingale-decomp adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w820 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "doom_decomp": b.bench_doom_decomp_family(),
        "pcdt": b.bench_pcdt_family(),
        "special_sem": b.bench_special_sem_family(),
        "canonical_decomp": b.bench_canonical_decomp_family(),
        "sem_loc_char": b.bench_sem_loc_char_family(),
        "triplet_char": b.bench_triplet_char_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
