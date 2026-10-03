"""Wave-360 PDE-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w360 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "energy_method": b.bench_energy_method_family(),
        "maximum_principle": b.bench_maximum_principle_family(),
        "heat_kernel": b.bench_heat_kernel_family(),
        "wave_dalembert": b.bench_wave_dalembert_family(),
        "weak_solution": b.bench_weak_solution_family(),
        "fundamental_laplace": b.bench_fundamental_laplace_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
