"""Wave-499 Hodge-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w499 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hodge_decomp": b.bench_hodge_decomp_family(),
        "l2_hodge": b.bench_l2_hodge_family(),
        "mixed_hodge": b.bench_mixed_hodge_family(),
        "period_map": b.bench_period_map_family(),
        "vhs_polarized": b.bench_vhs_polarized_family(),
        "limit_mhs": b.bench_limit_mhs_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
