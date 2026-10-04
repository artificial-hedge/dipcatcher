"""Wave-366 algebraic-topology-3 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w366 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "singular_homology": b.bench_singular_homology_family(),
        "cw_complex": b.bench_cw_complex_family(),
        "spectral_seq_toy": b.bench_spectral_seq_toy_family(),
        "homotopy_group": b.bench_homotopy_group_family(),
        "excision": b.bench_excision_family(),
        "poincare_dual": b.bench_poincare_dual_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
