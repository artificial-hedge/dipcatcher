"""Wave-452 geometric-Langlands adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w452 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "d_module": b.bench_d_module_family(),
        "geometric_langlands": b.bench_geometric_langlands_family(),
        "hecke_eig": b.bench_hecke_eig_family(),
        "opers_g": b.bench_opers_g_family(),
        "ramified_l": b.bench_ramified_l_family(),
        "kernel_fun": b.bench_kernel_fun_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
