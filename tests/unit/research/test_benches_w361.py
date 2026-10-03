"""Wave-361 algebraic-topology-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w361 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_pi1": b.bench_homotopy_pi1_family(),
        "simplicial_homology": b.bench_simplicial_homology_family(),
        "chain_homotopy": b.bench_chain_homotopy_family(),
        "euler_homology": b.bench_euler_homology_family(),
        "degree_map": b.bench_degree_map_family(),
        "covering_lift": b.bench_covering_lift_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
