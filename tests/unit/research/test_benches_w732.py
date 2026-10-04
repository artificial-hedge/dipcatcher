"""Wave-732 Hall-algebra adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w732 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hall_algebra": b.bench_hall_algebra_family(),
        "ringel_hall": b.bench_ringel_hall_family(),
        "toen_hall": b.bench_toen_hall_family(),
        "lusztig_hall": b.bench_lusztig_hall_family(),
        "schiffmann_hall": b.bench_schiffmann_hall_family(),
        "joyce_hall": b.bench_joyce_hall_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
