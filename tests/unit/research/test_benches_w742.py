"""Wave-742 random-matrix adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w742 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "soshnikov_rmt": b.bench_soshnikov_rmt_family(),
        "erdos_yau": b.bench_erdos_yau_family(),
        "forrester_rmt": b.bench_forrester_rmt_family(),
        "mehta_rmt": b.bench_mehta_rmt_family(),
        "deift_rmt": b.bench_deift_rmt_family(),
        "johansson_rmt": b.bench_johansson_rmt_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
