"""Wave-624 formal-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w624 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "raynaud_formal": b.bench_raynaud_formal_family(),
        "formal_completion": b.bench_formal_completion_family(),
        "adic_formal": b.bench_adic_formal_family(),
        "formal_neighborhood": b.bench_formal_neighborhood_family(),
        "groth_existence": b.bench_groth_existence_family(),
        "algebraization": b.bench_algebraization_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
