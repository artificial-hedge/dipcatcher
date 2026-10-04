"""Wave-539 diophantine-approximation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w539 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dirichlet_approx": b.bench_dirichlet_approx_family(),
        "roth_thm2": b.bench_roth_thm2_family(),
        "continued_frac2": b.bench_continued_frac2_family(),
        "kronecker_thm": b.bench_kronecker_thm_family(),
        "liouville_number": b.bench_liouville_number_family(),
        "subspace_thm": b.bench_subspace_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
