"""Wave-545 Thurston-geometrization adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w545 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "thurston_geometrization": b.bench_thurston_geometrization_family(),
        "eight_geometries": b.bench_eight_geometries_family(),
        "seifert_fibered": b.bench_seifert_fibered_family(),
        "haken_mfd": b.bench_haken_mfd_family(),
        "jsj_decomp": b.bench_jsj_decomp_family(),
        "ricci_flow": b.bench_ricci_flow_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
