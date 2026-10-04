"""Wave-608 sheaf-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w608 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "etale_descent": b.bench_etale_descent_family(),
        "etale_morphism": b.bench_etale_morphism_family(),
        "fppf_site": b.bench_fppf_site_family(),
        "fpqc_site": b.bench_fpqc_site_family(),
        "ladic_sheaf": b.bench_ladic_sheaf_family(),
        "lisse_sheaf": b.bench_lisse_sheaf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
