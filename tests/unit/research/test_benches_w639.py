"""Wave-639 homotopy-16 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w639 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "homotopy_fiber2": b.bench_homotopy_fiber2_family(),
        "stable_htpy2": b.bench_stable_htpy2_family(),
        "finite_htpy": b.bench_finite_htpy_family(),
        "rational_spec": b.bench_rational_spec_family(),
        "finite_chromatic": b.bench_finite_chromatic_family(),
        "periodic_htpy": b.bench_periodic_htpy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
