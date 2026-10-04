"""Wave-531 bifurcation-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w531 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "saddle_node": b.bench_saddle_node_family(),
        "hopf_bif": b.bench_hopf_bif_family(),
        "period_doubling": b.bench_period_doubling_family(),
        "neimark_sacker": b.bench_neimark_sacker_family(),
        "bogdanov_takens": b.bench_bogdanov_takens_family(),
        "homoclinic_bif": b.bench_homoclinic_bif_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
