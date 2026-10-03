"""Wave-407 algebraic-geometry-8 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w407 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cech_cohom": b.bench_cech_cohom_family(),
        "serre_duality": b.bench_serre_duality_family(),
        "adjunction2": b.bench_adjunction2_family(),
        "scheme_fiber": b.bench_scheme_fiber_family(),
        "hilbert_scheme": b.bench_hilbert_scheme_family(),
        "flattening": b.bench_flattening_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
