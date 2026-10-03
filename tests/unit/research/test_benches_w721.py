"""Wave-721 mixed-motives adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w721 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "brown_motives": b.bench_brown_motives_family(),
        "mzc_motive": b.bench_mzc_motive_family(),
        "zeta_element": b.bench_zeta_element_family(),
        "mixed_elliptic": b.bench_mixed_elliptic_family(),
        "motivic_pi": b.bench_motivic_pi_family(),
        "beilinson_height": b.bench_beilinson_height_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
