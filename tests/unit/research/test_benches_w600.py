"""Wave-600 operad-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w600 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "a_infty_alg": b.bench_a_infty_alg_family(),
        "l_infty_alg": b.bench_l_infty_alg_family(),
        "koszul_duality": b.bench_koszul_duality_family(),
        "minimal_model_op": b.bench_minimal_model_op_family(),
        "operadic_bar": b.bench_operadic_bar_family(),
        "operad_cobar": b.bench_operad_cobar_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
