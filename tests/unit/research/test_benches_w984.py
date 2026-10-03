"""Wave-984 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w984 import (
    bench_ambiguity_fn_family,
    bench_feichtinger_alg_family,
    bench_gabor_frame_family,
    bench_modulation_space_family,
    bench_short_time_ft_family,
    bench_wigner_dist_family,
)

_FAMILY_BENCHES = [
    bench_modulation_space_family,
    bench_short_time_ft_family,
    bench_gabor_frame_family,
    bench_wigner_dist_family,
    bench_ambiguity_fn_family,
    bench_feichtinger_alg_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
