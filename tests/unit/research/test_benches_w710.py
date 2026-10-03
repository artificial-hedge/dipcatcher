"""Wave-710 higher-algebra-11 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w710 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "higher_algebra8": b.bench_higher_algebra8_family(),
        "operad_infty4": b.bench_operad_infty4_family(),
        "floyd_farey": b.bench_floyd_farey_family(),
        "operad_swiss3": b.bench_operad_swiss3_family(),
        "little_discs3": b.bench_little_discs3_family(),
        "operad_twisted": b.bench_operad_twisted_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
