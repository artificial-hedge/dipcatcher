"""Wave-863 domain-decomposition adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w863 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "schwarz_add": b.bench_schwarz_add_family(),
        "schwarz_mult": b.bench_schwarz_mult_family(),
        "coarse_correction": b.bench_coarse_correction_family(),
        "mortar_dd": b.bench_mortar_dd_family(),
        "feti_lite": b.bench_feti_lite_family(),
        "bddc_lite": b.bench_bddc_lite_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
