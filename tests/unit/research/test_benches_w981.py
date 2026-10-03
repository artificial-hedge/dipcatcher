"""Wave-981 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w981 import (
    bench_grothendieck_const_family,
    bench_john_ellipsoid_family,
    bench_kadison_singer_family,
    bench_loewner_ellipsoid_family,
    bench_milman_isotropic_family,
    bench_milman_rev_thm_family,
)

_FAMILY_BENCHES = [
    bench_john_ellipsoid_family,
    bench_loewner_ellipsoid_family,
    bench_milman_rev_thm_family,
    bench_grothendieck_const_family,
    bench_kadison_singer_family,
    bench_milman_isotropic_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
