"""Wave-759 CLE-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w759 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gwynne_cle": b.bench_gwynne_cle_family(),
        "hospitsky_cle": b.bench_hospitsky_cle_family(),
        "apu_cle": b.bench_apu_cle_family(),
        "nolin_cle": b.bench_nolin_cle_family(),
        "sun_cle": b.bench_sun_cle_family(),
        "zhan_cle": b.bench_zhan_cle_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
