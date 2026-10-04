"""Wave-558 gauge-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w558 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "yang_mills": b.bench_yang_mills_family(),
        "instanton_moduli": b.bench_instanton_moduli_family(),
        "anti_self_dual": b.bench_anti_self_dual_family(),
        "higgs_bundle": b.bench_higgs_bundle_family(),
        "kapustin_witten": b.bench_kapustin_witten_family(),
        "nahm_transform": b.bench_nahm_transform_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
