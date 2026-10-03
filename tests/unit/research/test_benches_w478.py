"""Wave-478 motivic-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w478 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "levine_morel": b.bench_levine_morel_family(),
        "quadratic_k": b.bench_quadratic_k_family(),
        "mgl_spec": b.bench_mgl_spec_family(),
        "cellular_motive": b.bench_cellular_motive_family(),
        "motivic_pi0": b.bench_motivic_pi0_family(),
        "beilinson_con": b.bench_beilinson_con_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
