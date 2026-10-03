"""Wave-347 topology-3 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w347 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "compact_space": b.bench_compact_space_family(),
        "connected_space": b.bench_connected_space_family(),
        "quotient_topology": b.bench_quotient_topology_family(),
        "product_topology": b.bench_product_topology_family(),
        "convergence_space": b.bench_convergence_space_family(),
        "tietze_urysohn": b.bench_tietze_urysohn_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
