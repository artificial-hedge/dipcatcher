"""Wave-837 convex-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w837 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "brunn_minkowski": b.bench_brunn_minkowski_family(),
        "alexandrov_fenchel": b.bench_alexandrov_fenchel_family(),
        "isoperimetric_ineq": b.bench_isoperimetric_ineq_family(),
        "minkowski_sum": b.bench_minkowski_sum_family(),
        "mixed_volume": b.bench_mixed_volume_family(),
        "helly_theorem": b.bench_helly_theorem_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
