"""Wave-857 classical-quadrature adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w857 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gauss_legendre": b.bench_gauss_legendre_family(),
        "gauss_chebyshev": b.bench_gauss_chebyshev_family(),
        "clenshaw_curtis": b.bench_clenshaw_curtis_family(),
        "newton_cotes": b.bench_newton_cotes_family(),
        "gauss_kronrod": b.bench_gauss_kronrod_family(),
        "fejer_quad": b.bench_fejer_quad_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
