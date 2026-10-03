"""Wave-925 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w925 import (
    bench_bondesson_shot_family,
    bench_exchangeable_pf_family,
    bench_kingman_paintbox_family,
    bench_nggp_process_family,
    bench_normalized_rm_family,
    bench_sigma_stable_family,
)

_FAMILY_BENCHES = [
    bench_exchangeable_pf_family,
    bench_normalized_rm_family,
    bench_sigma_stable_family,
    bench_nggp_process_family,
    bench_bondesson_shot_family,
    bench_kingman_paintbox_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
