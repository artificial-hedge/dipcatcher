"""Wave-1191 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1191 import (
    bench_advertising_studies_family,
    bench_broadcasting_studies_family,
    bench_journalism_studies_family,
    bench_news_media_family,
    bench_public_relations_studies_family,
    bench_publishing_studies_family,
)

_FAMILY_BENCHES = [
    bench_journalism_studies_family,
    bench_advertising_studies_family,
    bench_broadcasting_studies_family,
    bench_news_media_family,
    bench_public_relations_studies_family,
    bench_publishing_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
