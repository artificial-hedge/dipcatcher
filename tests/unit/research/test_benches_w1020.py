"""Wave-1020 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1020 import (
    bench_food_web_family,
    bench_island_biogeography_family,
    bench_logistic_growth_family,
    bench_lotka_volterra_family,
    bench_neutral_theory_family,
    bench_predator_prey_family,
)

_FAMILY_BENCHES = [
    bench_predator_prey_family,
    bench_lotka_volterra_family,
    bench_logistic_growth_family,
    bench_island_biogeography_family,
    bench_neutral_theory_family,
    bench_food_web_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
