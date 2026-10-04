"""Wave-961 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w961 import (
    bench_analytic_semigroup_family,
    bench_c0_semigroup_family,
    bench_cosine_family_family,
    bench_hille_yosida_family,
    bench_lumer_phillips_family,
    bench_trotter_kato_family,
)

_FAMILY_BENCHES = [
    bench_c0_semigroup_family,
    bench_hille_yosida_family,
    bench_lumer_phillips_family,
    bench_analytic_semigroup_family,
    bench_cosine_family_family,
    bench_trotter_kato_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
