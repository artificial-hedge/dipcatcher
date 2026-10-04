"""Wave-1005 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1005 import (
    bench_einstein_equations_family,
    bench_friedmann_eq_family,
    bench_gr_birkhoff_family,
    bench_kerr_metric_family,
    bench_penrose_diagrams_family,
    bench_schwarzschild_metric_family,
)

_FAMILY_BENCHES = [
    bench_einstein_equations_family,
    bench_schwarzschild_metric_family,
    bench_friedmann_eq_family,
    bench_kerr_metric_family,
    bench_gr_birkhoff_family,
    bench_penrose_diagrams_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
