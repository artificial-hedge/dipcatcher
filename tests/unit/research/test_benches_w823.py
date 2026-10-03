"""Wave-823 law-of-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w823 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "support_law": b.bench_support_law_family(),
        "polish_law": b.bench_polish_law_family(),
        "tight_law": b.bench_tight_law_family(),
        "law_convergence": b.bench_law_convergence_family(),
        "finite_dim": b.bench_finite_dim_family(),
        "cylindrical_law": b.bench_cylindrical_law_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
