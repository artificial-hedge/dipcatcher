"""Wave-417 commutative-algebra-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w417 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "groebner_syz": b.bench_groebner_syz_family(),
        "free_resolution": b.bench_free_resolution_family(),
        "hilbert_syzygy": b.bench_hilbert_syzygy_family(),
        "regular_seq": b.bench_regular_seq_family(),
        "depth_ring": b.bench_depth_ring_family(),
        "cohen_mac": b.bench_cohen_mac_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
