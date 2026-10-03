"""Wave-913 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w913 import (
    bench_akima_interp_family,
    bench_makima_interp_family,
    bench_monotone_interp_family,
    bench_pchip_interp_family,
    bench_scattered_interp_family,
    bench_spline_interp_family,
)

_FAMILY_BENCHES = [
    bench_scattered_interp_family,
    bench_spline_interp_family,
    bench_monotone_interp_family,
    bench_akima_interp_family,
    bench_pchip_interp_family,
    bench_makima_interp_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
