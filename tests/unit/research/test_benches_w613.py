"""Wave-613 arithmetic-geometry-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w613 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "witt_vector": b.bench_witt_vector_family(),
        "witt_teich": b.bench_witt_teich_family(),
        "verschiebung_witt": b.bench_verschiebung_witt_family(),
        "perfect_witt": b.bench_perfect_witt_family(),
        "neron_smooth": b.bench_neron_smooth_family(),
        "semistable_reduction": b.bench_semistable_reduction_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
