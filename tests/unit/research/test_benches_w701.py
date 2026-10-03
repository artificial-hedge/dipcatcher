"""Wave-701 derived-geometry-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w701 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_conn": b.bench_derived_conn_family(),
        "derived_local": b.bench_derived_local_family(),
        "derived_reduced": b.bench_derived_reduced_family(),
        "derived_integral": b.bench_derived_integral_family(),
        "derived_normal": b.bench_derived_normal_family(),
        "derived_noether": b.bench_derived_noether_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
