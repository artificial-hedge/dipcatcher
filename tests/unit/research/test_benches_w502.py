"""Wave-502 dg-category adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w502 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dg_cat2": b.bench_dg_cat2_family(),
        "dg_morita": b.bench_dg_morita_family(),
        "dg_quotient": b.bench_dg_quotient_family(),
        "drinfeld_quotient": b.bench_drinfeld_quotient_family(),
        "dg_nerve": b.bench_dg_nerve_family(),
        "keller_dg": b.bench_keller_dg_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
