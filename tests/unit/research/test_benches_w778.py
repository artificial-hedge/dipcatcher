"""Wave-778 loss/vacation-queue adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w778 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "engset": b.bench_engset_family(),
        "erlang_b": b.bench_erlang_b_family(),
        "erlang_c": b.bench_erlang_c_family(),
        "pollaczek_khinchine": b.bench_pollaczek_khinchine_family(),
        "borel_tanner": b.bench_borel_tanner_family(),
        "takacs_vacation": b.bench_takacs_vacation_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
