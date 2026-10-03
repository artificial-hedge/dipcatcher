"""Wave-455 synthetic-math adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w455 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cubical_path": b.bench_cubical_path_family(),
        "hcomp_fill": b.bench_hcomp_fill_family(),
        "glue_types": b.bench_glue_types_family(),
        "interval_obj": b.bench_interval_obj_family(),
        "kan_op": b.bench_kan_op_family(),
        "transport_coe": b.bench_transport_coe_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
