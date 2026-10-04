"""Wave-434 Langlands-toy adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w434 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "satake_iso": b.bench_satake_iso_family(),
        "hecke_operator": b.bench_hecke_operator_family(),
        "langlands_dual": b.bench_langlands_dual_family(),
        "eisenstein_srs": b.bench_eisenstein_srs_family(),
        "automorphic_rep": b.bench_automorphic_rep_family(),
        "fourier_coeff": b.bench_fourier_coeff_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
