"""Wave-602 cyclic-homology adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w602 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cyclotomic_spec": b.bench_cyclotomic_spec_family(),
        "tr_structure": b.bench_tr_structure_family(),
        "tc_spec": b.bench_tc_spec_family(),
        "negative_cyclic": b.bench_negative_cyclic_family(),
        "periodic_cyclic": b.bench_periodic_cyclic_family(),
        "tate_construction": b.bench_tate_construction_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
