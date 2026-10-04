"""Wave-1060 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1060 import (
    bench_assessment_theory_family,
    bench_curriculum_design_family,
    bench_educational_psychology_family,
    bench_educational_technology_family,
    bench_learning_sciences_family,
    bench_pedagogy_family,
)

_FAMILY_BENCHES = [
    bench_curriculum_design_family,
    bench_pedagogy_family,
    bench_educational_psychology_family,
    bench_assessment_theory_family,
    bench_learning_sciences_family,
    bench_educational_technology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
