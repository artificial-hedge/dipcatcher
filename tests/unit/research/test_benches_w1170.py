"""Wave-1170 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1170 import (
    bench_criminology_3_family,
    bench_forensic_science_2_family,
    bench_intelligence_studies_2_family,
    bench_penology_2_family,
    bench_security_studies_2_family,
    bench_victimology_2_family,
)

_FAMILY_BENCHES = [
    bench_criminology_3_family,
    bench_forensic_science_2_family,
    bench_penology_2_family,
    bench_victimology_2_family,
    bench_security_studies_2_family,
    bench_intelligence_studies_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
