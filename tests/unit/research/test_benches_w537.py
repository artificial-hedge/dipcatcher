"""Wave-537 Riemannian-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w537 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "riemann_metric": b.bench_riemann_metric_family(),
        "levi_civita": b.bench_levi_civita_family(),
        "riemann_curvature": b.bench_riemann_curvature_family(),
        "ricci_scalar": b.bench_ricci_scalar_family(),
        "jacobi_field": b.bench_jacobi_field_family(),
        "comparison_thm": b.bench_comparison_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
