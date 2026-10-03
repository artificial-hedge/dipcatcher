"""Wave-461 DAG-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w461 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "shifted_sympl": b.bench_shifted_sympl_family(),
        "lagrangian_int": b.bench_lagrangian_int_family(),
        "derived_critical": b.bench_derived_critical_family(),
        "lie_algebroid": b.bench_lie_algebroid_family(),
        "moment_map": b.bench_moment_map_family(),
        "quant_dag": b.bench_quant_dag_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
