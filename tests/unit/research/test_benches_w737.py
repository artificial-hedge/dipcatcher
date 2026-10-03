"""Wave-737 LQG-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w737 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sheffield_quantum": b.bench_sheffield_quantum_family(),
        "gaines_sle": b.bench_gaines_sle_family(),
        "miller_wu": b.bench_miller_wu_family(),
        "rhoade_vargas": b.bench_rhoade_vargas_family(),
        "ding_dupias": b.bench_ding_dupias_family(),
        "gwynne_miller": b.bench_gwynne_miller_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
