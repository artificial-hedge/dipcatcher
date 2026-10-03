"""Wave-329 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w329 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "nd_check": b.bench_nd_check_family(),
        "sequent_prove": b.bench_sequent_prove_family(),
        "cut_elim": b.bench_cut_elim_family(),
        "resolution_fol": b.bench_resolution_fol_family(),
        "linear_logic": b.bench_linear_logic_family(),
        "intuit_class": b.bench_intuit_class_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
