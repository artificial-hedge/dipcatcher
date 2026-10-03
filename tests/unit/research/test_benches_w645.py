"""Wave-645 algebraic-K-9 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w645 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "witt_k": b.bench_witt_k_family(),
        "schlichting_k": b.bench_schlichting_k_family(),
        "balmer_k": b.bench_balmer_k_family(),
        "hermitian_k3": b.bench_hermitian_k3_family(),
        "thomason_les": b.bench_thomason_les_family(),
        "vishik_k": b.bench_vishik_k_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
