"""Wave-275 adapter bench tests."""

from quant_fund.research.benches_w275 import (
    bench_columnar_scan_family,
    bench_graceful_hash_family,
    bench_index_intersect_family,
    bench_late_materialize_family,
    bench_radix_join_family,
    bench_simd_filter_family,
)

FAMS = [
    bench_columnar_scan_family,
    bench_simd_filter_family,
    bench_late_materialize_family,
    bench_radix_join_family,
    bench_graceful_hash_family,
    bench_index_intersect_family,
]


def test_wave275_benches_all_synthetic() -> None:
    for f in FAMS:
        out = f()
        assert out, f.__name__
        for k in out:
            assert k.startswith("synthetic_"), k


def test_wave275_benches_score_high() -> None:
    for f in FAMS:
        assert max(f().values()) >= 0.5, f.__name__
