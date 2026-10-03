"""Wave-523 incidence-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w523 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "erdos_distinct": b.bench_erdos_distinct_family(),
        "sz_trotter": b.bench_sz_trotter_family(),
        "kakeya": b.bench_kakeya_family(),
        "ff_kakeya": b.bench_ff_kakeya_family(),
        "joints_thm": b.bench_joints_thm_family(),
        "guth_katz": b.bench_guth_katz_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
