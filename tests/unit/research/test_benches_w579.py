"""Wave-579 GIT adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w579 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "git_quotient": b.bench_git_quotient_family(),
        "hilbert_mumford": b.bench_hilbert_mumford_family(),
        "moment_polytope": b.bench_moment_polytope_family(),
        "kirwan_strat": b.bench_kirwan_strat_family(),
        "symplectic_quot": b.bench_symplectic_quot_family(),
        "luna_slice": b.bench_luna_slice_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
