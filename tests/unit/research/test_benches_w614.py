"""Wave-614 homotopy-13 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w614 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "primary_op": b.bench_primary_op_family(),
        "secondary_op": b.bench_secondary_op_family(),
        "steenrod_sq": b.bench_steenrod_sq_family(),
        "peterson_stein": b.bench_peterson_stein_family(),
        "moore_spec": b.bench_moore_spec_family(),
        "finite_spectra": b.bench_finite_spectra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
