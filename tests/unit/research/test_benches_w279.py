"""Wave-279 adapter bench tests."""

from quant_fund.research.benches_w279 import (
    bench_dist_obsv_family,
    bench_flat_track_family,
    bench_l2_gain_family,
    bench_luen_obsv_family,
    bench_lyap_synth_family,
    bench_mrac_adapt_family,
)

FAMS = [
    bench_luen_obsv_family,
    bench_dist_obsv_family,
    bench_mrac_adapt_family,
    bench_flat_track_family,
    bench_lyap_synth_family,
    bench_l2_gain_family,
]


def test_wave279_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave279_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
