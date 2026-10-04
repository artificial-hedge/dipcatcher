"""Wave-1082 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1082 import (
    bench_hebrew_language_family,
    bench_jewish_philosophy_family,
    bench_jewish_studies_family,
    bench_kabbalah_family,
    bench_rabbinics_family,
    bench_talmudic_studies_family,
)

_FAMILY_BENCHES = [
    bench_jewish_studies_family,
    bench_talmudic_studies_family,
    bench_hebrew_language_family,
    bench_rabbinics_family,
    bench_kabbalah_family,
    bench_jewish_philosophy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
