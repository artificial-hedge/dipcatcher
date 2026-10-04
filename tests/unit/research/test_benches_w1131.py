"""Wave-1131 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1131 import (
    bench_digital_sociology_family,
    bench_sociology_of_disaster_family,
    bench_sociology_of_housing_family,
    bench_sociology_of_migration_family,
    bench_sociology_of_risk_family,
    bench_sociology_of_the_body_family,
)

_FAMILY_BENCHES = [
    bench_sociology_of_migration_family,
    bench_sociology_of_housing_family,
    bench_sociology_of_disaster_family,
    bench_sociology_of_the_body_family,
    bench_sociology_of_risk_family,
    bench_digital_sociology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
