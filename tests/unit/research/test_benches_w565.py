"""Wave-565 arithmetic-statistics adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w565 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bhargava_lic": b.bench_bhargava_lic_family(),
        "cohen_lenstra": b.bench_cohen_lenstra_family(),
        "elliptic_rank": b.bench_elliptic_rank_family(),
        "malle_conj": b.bench_malle_conj_family(),
        "prime_gaps": b.bench_prime_gaps_family(),
        "zhang_maynard": b.bench_zhang_maynard_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
