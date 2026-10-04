"""Wave-1253 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1253 import (
    bench_antibody_studies_family,
    bench_chemokine_studies_family,
    bench_complement_studies_family,
    bench_cytokine_studies_family,
    bench_interferon_studies_family,
    bench_lymphocyte_studies_family,
)

_FAMILY_BENCHES = [
    bench_cytokine_studies_family,
    bench_chemokine_studies_family,
    bench_interferon_studies_family,
    bench_complement_studies_family,
    bench_antibody_studies_family,
    bench_lymphocyte_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
