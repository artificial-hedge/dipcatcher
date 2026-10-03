"""Wave-282 adapter bench tests."""

from quant_fund.research.benches_w282 import (
    bench_gray_code_family,
    bench_inversion_count_family,
    bench_latin_square_family,
    bench_ramsey_bound_family,
    bench_stirling_count_family,
    bench_subset_sum_dp_family,
)

FAMS = [
    bench_subset_sum_dp_family,
    bench_stirling_count_family,
    bench_gray_code_family,
    bench_inversion_count_family,
    bench_ramsey_bound_family,
    bench_latin_square_family,
]


def test_wave282_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave282_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
