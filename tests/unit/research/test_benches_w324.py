"""Wave-324 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w324 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fourier_motzkin": b.bench_fourier_motzkin_family(),
        "banerjee_dep": b.bench_banerjee_dep_family(),
        "pluto_schedule": b.bench_pluto_schedule_family(),
        "tiling_legality": b.bench_tiling_legality_family(),
        "omega_test": b.bench_omega_test_family(),
        "vec_legality": b.bench_vec_legality_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
