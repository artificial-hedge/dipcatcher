"""Wave-686 higher-algebra-8 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w686 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "e4_algebra": b.bench_e4_algebra_family(),
        "centralizer_alg": b.bench_centralizer_alg_family(),
        "delooping2": b.bench_delooping2_family(),
        "factorization_hom2": b.bench_factorization_hom2_family(),
        "koszul_duality2": b.bench_koszul_duality2_family(),
        "braces_e4": b.bench_braces_e4_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
