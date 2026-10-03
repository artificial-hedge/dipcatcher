"""Wave-692 higher-algebra-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w692 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "e5_algebra": b.bench_e5_algebra_family(),
        "little_cubes2": b.bench_little_cubes2_family(),
        "swiss_cheese3": b.bench_swiss_cheese3_family(),
        "framed_discs": b.bench_framed_discs_family(),
        "factorization_hom3": b.bench_factorization_hom3_family(),
        "centralizer_alg2": b.bench_centralizer_alg2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
