"""Wave-650 homotopy-20 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w650 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bousfield_htpy": b.bench_bousfield_htpy_family(),
        "dror_htpy": b.bench_dror_htpy_family(),
        "kane_htpy": b.bench_kane_htpy_family(),
        "moore_htpy": b.bench_moore_htpy_family(),
        "neisendorfer_htpy": b.bench_neisendorfer_htpy_family(),
        "anick_htpy": b.bench_anick_htpy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
