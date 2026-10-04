"""Wave-567 algebraic-combinatorics adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w567 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "littlewood_richardson": b.bench_littlewood_richardson_family(),
        "knuth_rsk": b.bench_knuth_rsk_family(),
        "macdonald_poly": b.bench_macdonald_poly_family(),
        "schubert_calc": b.bench_schubert_calc_family(),
        "honeycomb_tiling": b.bench_honeycomb_tiling_family(),
        "berenstein_zelevinsky": b.bench_berenstein_zelevinsky_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
