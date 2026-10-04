"""Wave-728 motivic-A1 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w728 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "emerton_glass": b.bench_emerton_glass_family(),
        "luan_yao": b.bench_luan_yao_family(),
        "morel_voev": b.bench_morel_voev_family(),
        "voev_homotopy": b.bench_voev_homotopy_family(),
        "totaro_cycle": b.bench_totaro_cycle_family(),
        "a1_degrees": b.bench_a1_degrees_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
