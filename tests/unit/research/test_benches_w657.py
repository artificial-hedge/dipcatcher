"""Wave-657 homotopy-22 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w657 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ravenel_htpy": b.bench_ravenel_htpy_family(),
        "bousfield_period": b.bench_bousfield_period_family(),
        "snake_constr": b.bench_snake_constr_family(),
        "homotopy_cartesian": b.bench_homotopy_cartesian_family(),
        "p_local_htpy": b.bench_p_local_htpy_family(),
        "completion_htpy": b.bench_completion_htpy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
