"""Wave-276 adapter bench tests."""

from quant_fund.research.benches_w276 import (
    bench_bully_elect_family,
    bench_causal_bcast_family,
    bench_chord_look_family,
    bench_quorum_rw_family,
    bench_ra_mutex_family,
    bench_token_ring_family,
)

FAMS = [
    bench_ra_mutex_family,
    bench_token_ring_family,
    bench_bully_elect_family,
    bench_chord_look_family,
    bench_quorum_rw_family,
    bench_causal_bcast_family,
]


def test_wave276_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave276_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
