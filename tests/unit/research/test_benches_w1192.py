"""Wave-1192 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1192 import (
    bench_acupuncture_studies_family,
    bench_chiropractic_studies_family,
    bench_herbal_medicine_family,
    bench_homeopathy_family,
    bench_naturopathy_family,
    bench_osteopathy_studies_family,
)

_FAMILY_BENCHES = [
    bench_acupuncture_studies_family,
    bench_chiropractic_studies_family,
    bench_naturopathy_family,
    bench_homeopathy_family,
    bench_herbal_medicine_family,
    bench_osteopathy_studies_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
