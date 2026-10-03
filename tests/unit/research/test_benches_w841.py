"""Wave-841 orthogonal-polynomial adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w841 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "legendre_poly": b.bench_legendre_poly_family(),
        "chebyshev_t": b.bench_chebyshev_t_family(),
        "hermite_poly": b.bench_hermite_poly_family(),
        "laguerre_poly": b.bench_laguerre_poly_family(),
        "jacobi_poly": b.bench_jacobi_poly_family(),
        "gegenbauer_poly": b.bench_gegenbauer_poly_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
