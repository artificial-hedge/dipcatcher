"""Wave-699 homotopy-29 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w699 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_general": b.bench_homotopy_general_family(),
        "homotopy_rational": b.bench_homotopy_rational_family(),
        "stable_dual": b.bench_stable_dual_family(),
        "stable_lie": b.bench_stable_lie_family(),
        "stable_motivic": b.bench_stable_motivic_family(),
        "stable_perf": b.bench_stable_perf_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
