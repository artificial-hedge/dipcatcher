"""Wave-424 set-theory-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w424 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "club_set": b.bench_club_set_family(),
        "stationary_set": b.bench_stationary_set_family(),
        "ultrafilter_toy": b.bench_ultrafilter_toy_family(),
        "partition_calc": b.bench_partition_calc_family(),
        "closed_unbounded": b.bench_closed_unbounded_family(),
        "mahlo_cardinal": b.bench_mahlo_cardinal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
