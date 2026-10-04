"""Wave-395 combinatorial-enumeration adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w395 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "catalan_dp": b.bench_catalan_dp_family(),
        "stirling_cycle": b.bench_stirling_cycle_family(),
        "partition_count": b.bench_partition_count_family(),
        "bell_triangle": b.bench_bell_triangle_family(),
        "eulerian_num": b.bench_eulerian_num_family(),
        "inclusion_excl": b.bench_inclusion_excl_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
