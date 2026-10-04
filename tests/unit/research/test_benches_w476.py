"""Wave-476 chromatic-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w476 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bp_spectrum": b.bench_bp_spectrum_family(),
        "adams_novikov": b.bench_adams_novikov_family(),
        "landweber_exact": b.bench_landweber_exact_family(),
        "greek_letter": b.bench_greek_letter_family(),
        "smith_toda": b.bench_smith_toda_family(),
        "picard_grp": b.bench_picard_grp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
