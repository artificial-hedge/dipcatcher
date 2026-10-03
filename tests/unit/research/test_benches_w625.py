"""Wave-625 homotopy-15 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w625 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "stable_cohomology2": b.bench_stable_cohomology2_family(),
        "woodward_op": b.bench_woodward_op_family(),
        "spectrum_type": b.bench_spectrum_type_family(),
        "complexity_spectrum": b.bench_complexity_spectrum_family(),
        "small_spec": b.bench_small_spec_family(),
        "simplicial_htpy": b.bench_simplicial_htpy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
