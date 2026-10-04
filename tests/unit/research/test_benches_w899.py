"""Wave-899 interpolation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w899 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cardinal_interp": b.bench_cardinal_interp_family(),
        "bernstein_form": b.bench_bernstein_form_family(),
        "shanks_trans": b.bench_shanks_trans_family(),
        "chebyshev_interp": b.bench_chebyshev_interp_family(),
        "osculating_interp": b.bench_osculating_interp_family(),
        "rational_interp": b.bench_rational_interp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
