"""Wave-477 arithmetic-D-modules-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w477 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dagger_dm": b.bench_dagger_dm_family(),
        "spencer_dm": b.bench_spencer_dm_family(),
        "caro_dm": b.bench_caro_dm_family(),
        "berthelot_rigid": b.bench_berthelot_rigid_family(),
        "berthelo_crys": b.bench_berthelo_crys_family(),
        "arithmetic_ht": b.bench_arithmetic_ht_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
