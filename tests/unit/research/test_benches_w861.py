"""Wave-861 time-marching/ODE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w861 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "imex_rk": b.bench_imex_rk_family(),
        "ssp_rk": b.bench_ssp_rk_family(),
        "exponential_euler": b.bench_exponential_euler_family(),
        "rosenbrock_w": b.bench_rosenbrock_w_family(),
        "ars_imex": b.bench_ars_imex_family(),
        "dirk_scheme": b.bench_dirk_scheme_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
