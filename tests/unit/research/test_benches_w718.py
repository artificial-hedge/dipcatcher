"""Wave-718 representation-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w718 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "helix_theory": b.bench_helix_theory_family(),
        "mutation_class": b.bench_mutation_class_family(),
        "rep_finite": b.bench_rep_finite_family(),
        "der_bimodule": b.bench_der_bimodule_family(),
        "icy_paper": b.bench_icy_paper_family(),
        "higher_auslander": b.bench_higher_auslander_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
