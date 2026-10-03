"""Wave-653 arithmetic-geometry-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w653 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fargues_scholze3": b.bench_fargues_scholze3_family(),
        "integral_padic2": b.bench_integral_padic2_family(),
        "ainf_cohom": b.bench_ainf_cohom_family(),
        "period_ring": b.bench_period_ring_family(),
        "galois_padic": b.bench_galois_padic_family(),
        "hodge_tate_padic": b.bench_hodge_tate_padic_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
