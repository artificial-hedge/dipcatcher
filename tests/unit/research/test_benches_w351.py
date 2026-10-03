"""Wave-351 functional-analysis adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w351 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "banach_fixed": b.bench_banach_fixed_family(),
        "spectral_theorem": b.bench_spectral_theorem_family(),
        "lp_duality": b.bench_lp_duality_family(),
        "fourier_finite": b.bench_fourier_finite_family(),
        "compact_operator": b.bench_compact_operator_family(),
        "gram_schmidt": b.bench_gram_schmidt_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
