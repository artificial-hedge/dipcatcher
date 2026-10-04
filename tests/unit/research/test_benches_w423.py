"""Wave-423 homological-algebra-4 adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w423 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "groth_spectral": b.bench_groth_spectral_family(),
        "serre_ss2": b.bench_serre_ss2_family(),
        "hypercohom": b.bench_hypercohom_family(),
        "deriv_hom": b.bench_deriv_hom_family(),
        "cartan_eilenberg": b.bench_cartan_eilenberg_family(),
        "adams_diff": b.bench_adams_diff_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
