"""Wave-604 homotopy-12 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w604 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "unstable_cohomology": b.bench_unstable_cohomology_family(),
        "may_ss": b.bench_may_ss_family(),
        "bokstedt_periodicity": b.bench_bokstedt_periodicity_family(),
        "topo_k_theory": b.bench_topo_k_theory_family(),
        "elliptic_k": b.bench_elliptic_k_family(),
        "equivariant_cohomology2": b.bench_equivariant_cohomology2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
