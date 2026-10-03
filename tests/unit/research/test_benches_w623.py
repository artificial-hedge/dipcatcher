"""Wave-623 higher-operads adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w623 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dendroidal2": b.bench_dendroidal2_family(),
        "operadic_nerve": b.bench_operadic_nerve_family(),
        "infty_operad2": b.bench_infty_operad2_family(),
        "a_infinity2": b.bench_a_infinity2_family(),
        "e_infinity3": b.bench_e_infinity3_family(),
        "cyclic_operad": b.bench_cyclic_operad_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
