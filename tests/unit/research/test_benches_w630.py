"""Wave-630 infinity-categories-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w630 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "quasi_cat2": b.bench_quasi_cat2_family(),
        "inner_horn": b.bench_inner_horn_family(),
        "joyal_horn": b.bench_joyal_horn_family(),
        "fib_infty": b.bench_fib_infty_family(),
        "cartesian_morphism": b.bench_cartesian_morphism_family(),
        "infty_functor": b.bench_infty_functor_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
