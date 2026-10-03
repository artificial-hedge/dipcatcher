"""Wave-850 Riemann-solver adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w850 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "roe_solver": b.bench_roe_solver_family(),
        "hllc_solver": b.bench_hllc_solver_family(),
        "ausm_flux": b.bench_ausm_flux_family(),
        "lax_friedrichs": b.bench_lax_friedrichs_family(),
        "godunov_exact": b.bench_godunov_exact_family(),
        "osher_solver": b.bench_osher_solver_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
