"""Wave-791 SPDE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w791 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spde_heat": b.bench_spde_heat_family(),
        "stochastic_burgers": b.bench_stochastic_burgers_family(),
        "kpz_equation": b.bench_kpz_equation_family(),
        "doering_mueller": b.bench_doering_mueller_family(),
        "quasilinear_spde": b.bench_quasilinear_spde_family(),
        "paracontrolled_spde": b.bench_paracontrolled_spde_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
