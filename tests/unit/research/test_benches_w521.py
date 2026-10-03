"""Wave-521 analytic-NT adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w521 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "explicit_formula": b.bench_explicit_formula_family(),
        "zero_density": b.bench_zero_density_family(),
        "riemann_zeta": b.bench_riemann_zeta_family(),
        "dirichlet_l": b.bench_dirichlet_l_family(),
        "linnik_thm": b.bench_linnik_thm_family(),
        "chebyshev_bias": b.bench_chebyshev_bias_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
