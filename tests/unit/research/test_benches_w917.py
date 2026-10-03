"""Wave-917 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w917 import (
    bench_bucket_sort_family,
    bench_chained_hash_family,
    bench_linear_probe_family,
    bench_rand_access_list_family,
    bench_shell_sort_family,
    bench_skew_list_family,
)

_FAMILY_BENCHES = [
    bench_chained_hash_family,
    bench_linear_probe_family,
    bench_bucket_sort_family,
    bench_shell_sort_family,
    bench_rand_access_list_family,
    bench_skew_list_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
