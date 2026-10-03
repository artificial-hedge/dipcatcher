"""Wave-267 adapter bench tests."""

from quant_fund.research.benches_w267 import (
    bench_bank_conflict_family,
    bench_mem_coalesce_family,
    bench_occupancy_calc_family,
    bench_shared_mem_tile_family,
    bench_simt_divergence_family,
    bench_warp_scheduler_family,
)

FAMS = [
    bench_warp_scheduler_family,
    bench_simt_divergence_family,
    bench_bank_conflict_family,
    bench_mem_coalesce_family,
    bench_occupancy_calc_family,
    bench_shared_mem_tile_family,
]


def test_wave267_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave267_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
