"""Wave-873 domain-decomposition adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w873 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dd_partition": b.bench_dd_partition_family(),
        "baldding_dd": b.bench_baldding_dd_family(),
        "neumann_dd": b.bench_neumann_dd_family(),
        "feti_dp": b.bench_feti_dp_family(),
        "subspace_dd": b.bench_subspace_dd_family(),
        "asm_precond": b.bench_asm_precond_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
