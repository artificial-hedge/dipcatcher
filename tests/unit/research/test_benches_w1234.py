"""Wave-1234 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1234 import (
    bench_breast_medicine_family,
    bench_contraception_studies_family,
    bench_infertility_studies_family,
    bench_menopause_medicine_family,
    bench_pelvic_health_studies_family,
    bench_urogynecology_studies_family,
)

_FAMILY_BENCHES = [
    bench_menopause_medicine_family,
    bench_urogynecology_studies_family,
    bench_breast_medicine_family,
    bench_infertility_studies_family,
    bench_contraception_studies_family,
    bench_pelvic_health_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
