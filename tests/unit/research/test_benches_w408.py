"""Wave-408 representation-theory-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w408 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "schur_functor": b.bench_schur_functor_family(),
        "brauer_alg": b.bench_brauer_alg_family(),
        "hecke_alg": b.bench_hecke_alg_family(),
        "casimir_op": b.bench_casimir_op_family(),
        "weight_space": b.bench_weight_space_family(),
        "bz_category": b.bench_bz_category_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
