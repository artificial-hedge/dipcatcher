"""Wave-512 super-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w512 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "super_space": b.bench_super_space_family(),
        "super_manifold": b.bench_super_manifold_family(),
        "super_lie": b.bench_super_lie_family(),
        "odd_variables": b.bench_odd_variables_family(),
        "berezin_int": b.bench_berezin_int_family(),
        "super_scheme": b.bench_super_scheme_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
