"""Wave-734 SLE adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w734 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "osgood_schramm": b.bench_osgood_schramm_family(),
        "lawler_werner": b.bench_lawler_werner_family(),
        "werner_wilson": b.bench_werner_wilson_family(),
        "smirnov_parafermion": b.bench_smirnov_parafermion_family(),
        "garmadon_sle": b.bench_garmadon_sle_family(),
        "miller_sheffield": b.bench_miller_sheffield_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
