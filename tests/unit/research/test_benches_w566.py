"""Wave-566 L-functions adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w566 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "selberg_trace2": b.bench_selberg_trace2_family(),
        "zero_spacing": b.bench_zero_spacing_family(),
        "montgomery_pair": b.bench_montgomery_pair_family(),
        "gue_statistics": b.bench_gue_statistics_family(),
        "keating_snaith": b.bench_keating_snaith_family(),
        "rudnick_sarnak": b.bench_rudnick_sarnak_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
