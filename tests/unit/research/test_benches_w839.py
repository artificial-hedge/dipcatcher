"""Wave-839 approximation-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w839 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "jackson_direct": b.bench_jackson_direct_family(),
        "chebyshev_alternation": b.bench_chebyshev_alternation_family(),
        "kolmogorov_nwidth": b.bench_kolmogorov_nwidth_family(),
        "bernstein_poly": b.bench_bernstein_poly_family(),
        "markov_brothers": b.bench_markov_brothers_family(),
        "fourier_decay": b.bench_fourier_decay_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
