"""Wave-354 algebraic-geometry-4 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w354 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "sheaf_gluing": b.bench_sheaf_gluing_family(),
        "local_ring_zn": b.bench_local_ring_zn_family(),
        "dedekind_check": b.bench_dedekind_check_family(),
        "divisor_group": b.bench_divisor_group_family(),
        "genus_riemann": b.bench_genus_riemann_family(),
        "moduli_naive": b.bench_moduli_naive_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
