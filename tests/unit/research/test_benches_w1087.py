"""Wave-1087 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1087 import (
    bench_diplomatics_family,
    bench_epigraphy_family,
    bench_genealogy_studies_family,
    bench_heraldry_family,
    bench_onomastics_family,
    bench_sigillography_family,
)

_FAMILY_BENCHES = [
    bench_epigraphy_family,
    bench_diplomatics_family,
    bench_sigillography_family,
    bench_heraldry_family,
    bench_genealogy_studies_family,
    bench_onomastics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
