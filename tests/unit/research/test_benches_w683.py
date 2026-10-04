"""Wave-683 chromatic-8 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w683 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "blue_shift2": b.bench_blue_shift2_family(),
        "chromatic_fracture2": b.bench_chromatic_fracture2_family(),
        "morava_stabilizer2": b.bench_morava_stabilizer2_family(),
        "fgsl_group2": b.bench_fgsl_group2_family(),
        "tate_spec2": b.bench_tate_spec2_family(),
        "k_n_local2": b.bench_k_n_local2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
