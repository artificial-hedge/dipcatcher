"""Wave-380 descriptive-set-theory-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w380 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "baire_space": b.bench_baire_space_family(),
        "polish_topology": b.bench_polish_topology_family(),
        "borel_functions": b.bench_borel_functions_family(),
        "souslin_op": b.bench_souslin_op_family(),
        "determinacy_toy": b.bench_determinacy_toy_family(),
        "perfect_set_prop": b.bench_perfect_set_prop_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
