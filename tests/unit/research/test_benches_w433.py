"""Wave-433 p-adic-geometry adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w433 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "rigid_analytic": b.bench_rigid_analytic_family(),
        "berkovich_space": b.bench_berkovich_space_family(),
        "perfectoid_space": b.bench_perfectoid_space_family(),
        "adic_space": b.bench_adic_space_family(),
        "etale_ph2": b.bench_etale_ph2_family(),
        "diamond_toy": b.bench_diamond_toy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
