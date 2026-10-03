"""Wave-322 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w322 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "weakest_precond": b.bench_weakest_precond_family(),
        "sygus_synth": b.bench_sygus_synth_family(),
        "horn_clauses": b.bench_horn_clauses_family(),
        "interpolant_mc": b.bench_interpolant_mc_family(),
        "predicate_abs": b.bench_predicate_abs_family(),
        "cegis_loop": b.bench_cegis_loop_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
