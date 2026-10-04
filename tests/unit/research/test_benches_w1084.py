"""Wave-1084 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1084 import (
    bench_comparative_literature_family,
    bench_critical_theory_family,
    bench_literary_theory_family,
    bench_postcolonial_studies_family,
    bench_translation_studies_family,
    bench_world_literature_family,
)

_FAMILY_BENCHES = [
    bench_comparative_literature_family,
    bench_literary_theory_family,
    bench_postcolonial_studies_family,
    bench_world_literature_family,
    bench_translation_studies_family,
    bench_critical_theory_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
