"""Wave-582 geometric-Langlands-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w582 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "arinkin_gaitsgory": b.bench_arinkin_gaitsgory_family(),
        "derived_satake": b.bench_derived_satake_family(),
        "spectral_bung": b.bench_spectral_bung_family(),
        "nilp_cone": b.bench_nilp_cone_family(),
        "geometric_satake2": b.bench_geometric_satake2_family(),
        "fusion_product": b.bench_fusion_product_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
