"""Wave-470 homotopy-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w470 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "e_infty2": b.bench_e_infty2_family(),
        "power_op": b.bench_power_op_family(),
        "obstruction_th": b.bench_obstruction_th_family(),
        "rational_htpy": b.bench_rational_htpy_family(),
        "h_space": b.bench_h_space_family(),
        "james_constr": b.bench_james_constr_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
