"""Wave-383 homotopy-theory-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w383 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "mapping_cone": b.bench_mapping_cone_family(),
        "loop_space": b.bench_loop_space_family(),
        "em_space": b.bench_em_space_family(),
        "co_homology": b.bench_co_homology_family(),
        "stiefel_whitney": b.bench_stiefel_whitney_family(),
        "transfer": b.bench_transfer_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
