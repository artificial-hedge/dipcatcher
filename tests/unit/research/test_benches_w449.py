"""Wave-449 analytic-geometry-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w449 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dagger_space": b.bench_dagger_space_family(),
        "huber_ring": b.bench_huber_ring_family(),
        "adic_generic": b.bench_adic_generic_family(),
        "witt_perfect": b.bench_witt_perfect_family(),
        "fargues_curve": b.bench_fargues_curve_family(),
        "prism_site": b.bench_prism_site_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
