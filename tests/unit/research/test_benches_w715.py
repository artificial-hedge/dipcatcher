"""Wave-715 cluster-algebra adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w715 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cluster_algebra": b.bench_cluster_algebra_family(),
        "quiver_mutation": b.bench_quiver_mutation_family(),
        "tilting_object": b.bench_tilting_object_family(),
        "auslander_reiten": b.bench_auslander_reiten_family(),
        "cluster_category": b.bench_cluster_category_family(),
        "silting_object": b.bench_silting_object_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
