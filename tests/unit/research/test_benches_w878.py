"""Wave-878 low-rank/hierarchical-matrix adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w878 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "low_rank_svd": b.bench_low_rank_svd_family(),
        "h_matrix": b.bench_h_matrix_family(),
        "hss_matrix": b.bench_hss_matrix_family(),
        "randomized_nystrom": b.bench_randomized_nystrom_family(),
        "block_low_rank": b.bench_block_low_rank_family(),
        "kronecker_approx": b.bench_kronecker_approx_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
