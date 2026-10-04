"""Wave-787 regularity-structure adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w787 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ito_signature": b.bench_ito_signature_family(),
        "lyons_extension": b.bench_lyons_extension_family(),
        "tame_map": b.bench_tame_map_family(),
        "step_signature": b.bench_step_signature_family(),
        "gubinelli_sewing": b.bench_gubinelli_sewing_family(),
        "young_integral": b.bench_young_integral_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
