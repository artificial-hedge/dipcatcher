"""Wave-799 path-PDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w799 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "path_dependent_pde": b.bench_path_dependent_pde_family(),
        "functional_ito": b.bench_functional_ito_family(),
        "dupire_functional": b.bench_dupire_functional_family(),
        "viscosity_path": b.bench_viscosity_path_family(),
        "path_sobolev": b.bench_path_sobolev_family(),
        "kolmogorov_path": b.bench_kolmogorov_path_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
