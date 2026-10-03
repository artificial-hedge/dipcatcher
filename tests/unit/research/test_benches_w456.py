"""Wave-456 pure-motives adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w456 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "chow_motive": b.bench_chow_motive_family(),
        "nori_motive": b.bench_nori_motive_family(),
        "num_equiv": b.bench_num_equiv_family(),
        "standard_conj": b.bench_standard_conj_family(),
        "voev_motive": b.bench_voev_motive_family(),
        "tate_motive": b.bench_tate_motive_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
