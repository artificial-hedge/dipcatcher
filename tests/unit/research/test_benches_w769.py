"""Wave-769 Stein-method adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w769 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "stein_method": b.bench_stein_method_family(),
        "stein_equation": b.bench_stein_equation_family(),
        "barbour_stein": b.bench_barbour_stein_family(),
        "chen_stein": b.bench_chen_stein_family(),
        "ross_stein": b.bench_ross_stein_family(),
        "chatt_stein": b.bench_chatt_stein_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
