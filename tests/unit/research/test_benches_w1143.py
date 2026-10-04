"""Wave-1143 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1143 import (
    bench_acoustics_2_family,
    bench_biophysics_2_family,
    bench_condensed_matter_3_family,
    bench_nanotechnology_family,
    bench_optics_3_family,
    bench_thermodynamics_2_family,
)

_FAMILY_BENCHES = [
    bench_nanotechnology_family,
    bench_biophysics_2_family,
    bench_condensed_matter_3_family,
    bench_optics_3_family,
    bench_acoustics_2_family,
    bench_thermodynamics_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
