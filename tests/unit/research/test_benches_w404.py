"""Wave-404 operad-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w404 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "operad_algt": b.bench_operad_algt_family(),
        "brace_operad": b.bench_brace_operad_family(),
        "swiss_cheese": b.bench_swiss_cheese_family(),
        "little_intervals": b.bench_little_intervals_family(),
        "operad_homology": b.bench_operad_homology_family(),
        "props_toy": b.bench_props_toy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
