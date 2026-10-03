"""Wave-420 algebraic-NT-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w420 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dirichlet_unit": b.bench_dirichlet_unit_family(),
        "regulator": b.bench_regulator_family(),
        "ideal_class": b.bench_ideal_class_family(),
        "minkowski_bound": b.bench_minkowski_bound_family(),
        "dedekind_zeta": b.bench_dedekind_zeta_family(),
        "splitting_prime": b.bench_splitting_prime_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
