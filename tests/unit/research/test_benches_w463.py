"""Wave-463 arithmetic-D-modules adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w463 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "overconv_dm": b.bench_overconv_dm_family(),
        "arithmetic_dm": b.bench_arithmetic_dm_family(),
        "frobenius_dm": b.bench_frobenius_dm_family(),
        "holonomic_dm": b.bench_holonomic_dm_family(),
        "rigid_dm": b.bench_rigid_dm_family(),
        "isocrystal": b.bench_isocrystal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
