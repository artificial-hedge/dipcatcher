"""Wave-796 FBSDE-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w796 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "four_step_scheme": b.bench_four_step_scheme_family(),
        "decoupling_field2": b.bench_decoupling_field2_family(),
        "quasi_bsde": b.bench_quasi_bsde_family(),
        "coupled_fbsde": b.bench_coupled_fbsde_family(),
        "random_bsde": b.bench_random_bsde_family(),
        "time_bsde": b.bench_time_bsde_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
