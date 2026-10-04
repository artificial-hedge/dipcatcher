"""Wave-1069 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1069 import (
    bench_criminal_justice_family,
    bench_criminal_procedure_family,
    bench_forensic_science_family,
    bench_penology_family,
    bench_policing_studies_family,
    bench_victimology_family,
)

_FAMILY_BENCHES = [
    bench_criminal_justice_family,
    bench_forensic_science_family,
    bench_penology_family,
    bench_policing_studies_family,
    bench_victimology_family,
    bench_criminal_procedure_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
