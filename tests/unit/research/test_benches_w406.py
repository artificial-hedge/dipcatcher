"""Wave-406 homological-algebra-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w406 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "poincare_duality2": b.bench_poincare_duality2_family(),
        "universal_coeff": b.bench_universal_coeff_family(),
        "kunneth": b.bench_kunneth_family(),
        "leray_hirsch": b.bench_leray_hirsch_family(),
        "hopf_algebra2": b.bench_hopf_algebra2_family(),
        "functor_derived": b.bench_functor_derived_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
