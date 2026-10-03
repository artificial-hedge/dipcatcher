"""Wave-396 algebraic-number-theory-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w396 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cyclotomic_field": b.bench_cyclotomic_field_family(),
        "kronecker_weber": b.bench_kronecker_weber_family(),
        "local_field": b.bench_local_field_family(),
        "hensel_field": b.bench_hensel_field_family(),
        "cm_points": b.bench_cm_points_family(),
        "idele_class": b.bench_idele_class_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
