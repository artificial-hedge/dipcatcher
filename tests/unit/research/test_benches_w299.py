"""Adapter tests for wave-299 game-playing-2 canon benches."""

from quant_fund.research.benches_w299 import (
    bench_expectimax_family,
    bench_isomcts_family,
    bench_mast_playout_family,
    bench_rave_mc_family,
    bench_retrograde_wdl_family,
    bench_tablebase_dtm_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_tablebase_dtm_family,
        bench_retrograde_wdl_family,
        bench_rave_mc_family,
        bench_mast_playout_family,
        bench_expectimax_family,
        bench_isomcts_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
