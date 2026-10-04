"""Wave-1241 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1241 import (
    bench_anemia_studies_family,
    bench_bleeding_disorders_family,
    bench_coagulation_studies_family,
    bench_hemoglobin_studies_family,
    bench_marrow_studies_family,
    bench_thrombosis_medicine_family,
)

_FAMILY_BENCHES = [
    bench_anemia_studies_family,
    bench_coagulation_studies_family,
    bench_hemoglobin_studies_family,
    bench_thrombosis_medicine_family,
    bench_bleeding_disorders_family,
    bench_marrow_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
