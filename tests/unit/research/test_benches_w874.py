"""Wave-874 reliability-analysis adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w874 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "uq_reliability": b.bench_uq_reliability_family(),
        "first_order_rel": b.bench_first_order_rel_family(),
        "sorm_method": b.bench_sorm_method_family(),
        "subset_sim": b.bench_subset_sim_family(),
        "line_sampling": b.bench_line_sampling_family(),
        "metamodel_rel": b.bench_metamodel_rel_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
