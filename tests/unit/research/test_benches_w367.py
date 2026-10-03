"""Wave-367 Galois-2/field-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w367 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "finite_field": b.bench_finite_field_family(),
        "galois_corresp": b.bench_galois_corresp_family(),
        "normality_check": b.bench_normality_check_family(),
        "separable_check": b.bench_separable_check_family(),
        "cyclotomic_poly": b.bench_cyclotomic_poly_family(),
        "primitive_elem": b.bench_primitive_elem_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
