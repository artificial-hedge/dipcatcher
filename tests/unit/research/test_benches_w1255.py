"""Wave-1255 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1255 import (
    bench_als_studies_family,
    bench_alzheimer_studies_family,
    bench_dementia_studies_family,
    bench_huntington_studies_family,
    bench_ms_studies_family,
    bench_parkinson_studies_family,
)

_FAMILY_BENCHES = [
    bench_parkinson_studies_family,
    bench_alzheimer_studies_family,
    bench_ms_studies_family,
    bench_als_studies_family,
    bench_huntington_studies_family,
    bench_dementia_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
