"""Wave-980 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w980 import (
    bench_de_boor_stable_family,
    bench_faber_schauder_family,
    bench_haar_system_family,
    bench_korovkin_thm_family,
    bench_walsh_series_family,
    bench_whitney_ext_family,
)

_FAMILY_BENCHES = [
    bench_walsh_series_family,
    bench_haar_system_family,
    bench_faber_schauder_family,
    bench_de_boor_stable_family,
    bench_whitney_ext_family,
    bench_korovkin_thm_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
