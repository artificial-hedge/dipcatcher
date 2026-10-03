"""Adapter tests for wave-306 astronomy-3/IOD canon benches."""

from quant_fund.research.benches_w306 import (
    bench_batch_od_family,
    bench_cowell_j2_family,
    bench_cr3bp_dynamics_family,
    bench_davenport_q_family,
    bench_laplace_iod_family,
    bench_porkchop_grid_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_laplace_iod_family,
        bench_cowell_j2_family,
        bench_batch_od_family,
        bench_cr3bp_dynamics_family,
        bench_porkchop_grid_family,
        bench_davenport_q_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
