"""Wave-440 homotopy-8 adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w440 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "thom_iso": b.bench_thom_iso_family(),
        "postnikov_twr": b.bench_postnikov_twr_family(),
        "whitehead_twr": b.bench_whitehead_twr_family(),
        "bott_period": b.bench_bott_period_family(),
        "stable_stem": b.bench_stable_stem_family(),
        "hopf_map": b.bench_hopf_map_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
