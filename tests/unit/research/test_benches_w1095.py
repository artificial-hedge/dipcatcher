"""Wave-1095 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1095 import (
    bench_deviance_studies_family,
    bench_family_sociology_family,
    bench_medical_sociology_family,
    bench_organization_theory_family,
    bench_rural_sociology_family,
    bench_social_movements_family,
)

_FAMILY_BENCHES = [
    bench_medical_sociology_family,
    bench_deviance_studies_family,
    bench_family_sociology_family,
    bench_organization_theory_family,
    bench_social_movements_family,
    bench_rural_sociology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
