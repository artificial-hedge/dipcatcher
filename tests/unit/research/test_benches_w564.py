"""Wave-564 geometric-group-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w564 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gromov_hyperbolic": b.bench_gromov_hyperbolic_family(),
        "quasi_isometry": b.bench_quasi_isometry_family(),
        "thin_triangle": b.bench_thin_triangle_family(),
        "word_problem": b.bench_word_problem_family(),
        "baumslag_solitar": b.bench_baumslag_solitar_family(),
        "asymptotic_cone": b.bench_asymptotic_cone_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
