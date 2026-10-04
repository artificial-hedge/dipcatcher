"""Wave-743 random-matrix-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w743 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "baik_rmt": b.bench_baik_rmt_family(),
        "tao_vu": b.bench_tao_vu_family(),
        "borodin_olshanski": b.bench_borodin_olshanski_family(),
        "cipolloni_erdos": b.bench_cipolloni_erdos_family(),
        "bourgade_rmt": b.bench_bourgade_rmt_family(),
        "chafai_rmt": b.bench_chafai_rmt_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
