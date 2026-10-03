"""Wave-338 set-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w338 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ordinal_arith": b.bench_ordinal_arith_family(),
        "cardinal_arith": b.bench_cardinal_arith_family(),
        "transfinite_induct": b.bench_transfinite_induct_family(),
        "well_founded": b.bench_well_founded_family(),
        "v_omega": b.bench_v_omega_family(),
        "ac_choice": b.bench_ac_choice_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
