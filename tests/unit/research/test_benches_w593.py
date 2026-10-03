"""Wave-593 nonabelian-Hodge adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w593 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "higgs_bundle2": b.bench_higgs_bundle2_family(),
        "hitchin_section": b.bench_hitchin_section_family(),
        "simpson_corr": b.bench_simpson_corr_family(),
        "nonabelian_hodge": b.bench_nonabelian_hodge_family(),
        "harmonic_bdl": b.bench_harmonic_bdl_family(),
        "hodge_moduli": b.bench_hodge_moduli_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
