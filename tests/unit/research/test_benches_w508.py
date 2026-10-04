"""Wave-508 anabelian-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w508 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "anabelian_geo": b.bench_anabelian_geo_family(),
        "section_conj": b.bench_section_conj_family(),
        "fundamental_grp": b.bench_fundamental_grp_family(),
        "etale_pi1": b.bench_etale_pi1_family(),
        "groth_tei": b.bench_groth_tei_family(),
        "tamagawa_mochi": b.bench_tamagawa_mochi_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
