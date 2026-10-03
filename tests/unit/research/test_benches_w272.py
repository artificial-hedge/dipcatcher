"""Wave-272 adapter bench tests."""

from quant_fund.research.benches_w272 import (
    bench_backstepping_family,
    bench_gain_schedule_family,
    bench_pid_antiwindup_family,
    bench_repetitive_ctrl_family,
    bench_sliding_mode_family,
    bench_smith_predictor_family,
)

FAMS = [
    bench_pid_antiwindup_family,
    bench_sliding_mode_family,
    bench_gain_schedule_family,
    bench_smith_predictor_family,
    bench_backstepping_family,
    bench_repetitive_ctrl_family,
]


def test_wave272_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave272_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
