"""Wave-534 potential-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w534 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "harmonic_fn": b.bench_harmonic_fn_family(),
        "potential_thy": b.bench_potential_thy_family(),
        "capacity_theory": b.bench_capacity_theory_family(),
        "balayage": b.bench_balayage_family(),
        "green_fn": b.bench_green_fn_family(),
        "fine_topology": b.bench_fine_topology_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
