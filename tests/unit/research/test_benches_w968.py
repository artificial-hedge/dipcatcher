"""Wave-968 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w968 import (
    bench_baaj_julg_family,
    bench_cuntz_picture_family,
    bench_ext_functor_family,
    bench_kasparov_prod_family,
    bench_kk_duality_family,
    bench_kk_theory_family,
)

_FAMILY_BENCHES = [
    bench_kk_theory_family,
    bench_kasparov_prod_family,
    bench_ext_functor_family,
    bench_baaj_julg_family,
    bench_cuntz_picture_family,
    bench_kk_duality_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
