"""Wave-693 derived-geometry-8 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w693 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_etale": b.bench_derived_etale_family(),
        "derived_flat": b.bench_derived_flat_family(),
        "derived_smooth2": b.bench_derived_smooth2_family(),
        "derived_quasi_coherent": b.bench_derived_quasi_coherent_family(),
        "derived_represent": b.bench_derived_represent_family(),
        "derived_cartesian": b.bench_derived_cartesian_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
