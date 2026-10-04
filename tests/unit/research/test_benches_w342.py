"""Wave-342 descriptive-set-theory/recursion-2 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w342 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "borel_hierarchy": b.bench_borel_hierarchy_family(),
        "analytic_sets": b.bench_analytic_sets_family(),
        "forcing_lite": b.bench_forcing_lite_family(),
        "arith_hierarchy": b.bench_arith_hierarchy_family(),
        "jump_operator": b.bench_jump_operator_family(),
        "rice_theorem": b.bench_rice_theorem_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
