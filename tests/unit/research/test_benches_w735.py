"""Wave-735 SLE-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w735 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "beffara_sle": b.bench_beffara_sle_family(),
        "kemppainen_sle": b.bench_kemppainen_sle_family(),
        "zykin_sle": b.bench_zykin_sle_family(),
        "viklund_sle": b.bench_viklund_sle_family(),
        "benoist_sle": b.bench_benoist_sle_family(),
        "holden_sle": b.bench_holden_sle_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
