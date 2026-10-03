"""Wave-369 numerical-6 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w369 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "broyden": b.bench_broyden_family(),
        "cheb_approx": b.bench_cheb_approx_family(),
        "brent_root": b.bench_brent_root_family(),
        "romberg": b.bench_romberg_family(),
        "aitken_delta": b.bench_aitken_delta_family(),
        "collocation_ode": b.bench_collocation_ode_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
