"""Wave-756 GFF adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w756 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "berestycki_gff": b.bench_berestycki_gff_family(),
        "duplantier_sheffield": b.bench_duplantier_sheffield_family(),
        "houchmandzadeh_gff": b.bench_houchmandzadeh_gff_family(),
        "nick_gff": b.bench_nick_gff_family(),
        "sheffield_miller": b.bench_sheffield_miller_family(),
        "wiegmann_zabrodin": b.bench_wiegmann_zabrodin_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
