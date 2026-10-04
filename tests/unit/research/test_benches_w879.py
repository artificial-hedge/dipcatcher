"""Wave-879 exponential-time-integrator adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w879 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "expm_int": b.bench_expm_int_family(),
        "expokit": b.bench_expokit_family(),
        "krylov_subspace_time": b.bench_krylov_subspace_time_family(),
        "leja_point": b.bench_leja_point_family(),
        "phi_function": b.bench_phi_function_family(),
        "etd_rk4_classic": b.bench_etd_rk4_classic_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
