"""Wave-842 spectral-methods adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w842 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "chebyshev_grid": b.bench_chebyshev_grid_family(),
        "fourier_galerkin": b.bench_fourier_galerkin_family(),
        "legendre_tau": b.bench_legendre_tau_family(),
        "chebyshev_collocation": b.bench_chebyshev_collocation_family(),
        "spectral_deriv": b.bench_spectral_deriv_family(),
        "dealiasing": b.bench_dealiasing_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
