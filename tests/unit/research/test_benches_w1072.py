"""Wave-1072 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1072 import (
    bench_cinema_studies_family,
    bench_documentary_studies_family,
    bench_film_history_family,
    bench_film_studies_family,
    bench_film_theory_family,
    bench_screenwriting_family,
)

_FAMILY_BENCHES = [
    bench_film_studies_family,
    bench_cinema_studies_family,
    bench_film_theory_family,
    bench_film_history_family,
    bench_documentary_studies_family,
    bench_screenwriting_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
