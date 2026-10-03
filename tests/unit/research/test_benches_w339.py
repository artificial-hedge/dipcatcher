"""Wave-339 algebra adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w339 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "field_ext": b.bench_field_ext_family(),
        "galois_group": b.bench_galois_group_family(),
        "splitting_field": b.bench_splitting_field_family(),
        "lie_bracket": b.bench_lie_bracket_family(),
        "rep_theory": b.bench_rep_theory_family(),
        "root_system": b.bench_root_system_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
