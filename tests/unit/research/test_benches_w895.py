"""Wave-895 root-finding adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w895 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "secant_root": b.bench_secant_root_family(),
        "regula_falsi": b.bench_regula_falsi_family(),
        "muller_root": b.bench_muller_root_family(),
        "aitken_steffensen": b.bench_aitken_steffensen_family(),
        "richardson_limit": b.bench_richardson_limit_family(),
        "bulirsch_stoer": b.bench_bulirsch_stoer_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
