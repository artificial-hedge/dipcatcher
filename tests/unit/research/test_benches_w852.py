"""Wave-852 boundary-element adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w852 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bem_kernel": b.bench_bem_kernel_family(),
        "fredholm_solve": b.bench_fredholm_solve_family(),
        "nystrom_method": b.bench_nystrom_method_family(),
        "singular_integrals": b.bench_singular_integrals_family(),
        "fast_multipole": b.bench_fast_multipole_family(),
        "galerkin_bem": b.bench_galerkin_bem_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
