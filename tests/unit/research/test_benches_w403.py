"""Wave-403 homotopy-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w403 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "j_hom_toy": b.bench_j_hom_toy_family(),
        "toda_bracket": b.bench_toda_bracket_family(),
        "spectral_atiyah": b.bench_spectral_atiyah_family(),
        "pi_stems": b.bench_pi_stems_family(),
        "hopf_invariant": b.bench_hopf_invariant_family(),
        "thom_spectrum": b.bench_thom_spectrum_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
