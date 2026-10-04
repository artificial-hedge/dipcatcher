"""Wave-443 motivic-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w443 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "motivic_coh": b.bench_motivic_coh_family(),
        "chow_group": b.bench_chow_group_family(),
        "milnor_conj": b.bench_milnor_conj_family(),
        "voevodsky_dm": b.bench_voevodsky_dm_family(),
        "motivic_stem": b.bench_motivic_stem_family(),
        "brauer_grp": b.bench_brauer_grp_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
