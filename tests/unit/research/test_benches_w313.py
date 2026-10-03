"""Wave-313 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w313 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "v_cycle": b.bench_v_cycle_family(),
        "amg_lite": b.bench_amg_lite_family(),
        "bicgstab": b.bench_bicgstab_family(),
        "minres": b.bench_minres_family(),
        "chebyshev_iter": b.bench_chebyshev_iter_family(),
        "ilu_precond": b.bench_ilu_precond_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
