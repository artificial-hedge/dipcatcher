"""Wave-846 special-functions adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w846 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gamma_fn": b.bench_gamma_fn_family(),
        "beta_fn": b.bench_beta_fn_family(),
        "bessel_fn": b.bench_bessel_fn_family(),
        "airy_fn": b.bench_airy_fn_family(),
        "error_fn": b.bench_error_fn_family(),
        "hypergeometric_fn": b.bench_hypergeometric_fn_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
