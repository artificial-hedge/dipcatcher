"""Wave-773 queueing adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w773 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "mm1_queue": b.bench_mm1_queue_family(),
        "mg1_queue": b.bench_mg1_queue_family(),
        "gm_queue": b.bench_gm_queue_family(),
        "bulk_queue": b.bench_bulk_queue_family(),
        "retrial_queue": b.bench_retrial_queue_family(),
        "priority_queue": b.bench_priority_queue_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
