"""Wave-856 quadrature/quasi-MC adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w856 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "monte_carlo_quad": b.bench_monte_carlo_quad_family(),
        "quasi_mc": b.bench_quasi_mc_family(),
        "halton_seq": b.bench_halton_seq_family(),
        "sobol_seq": b.bench_sobol_seq_family(),
        "latin_hypercube": b.bench_latin_hypercube_family(),
        "stratified_mc": b.bench_stratified_mc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
