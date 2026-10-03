"""Wave-725 automorphic-points adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w725 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "p_group_iwasawa": b.bench_p_group_iwasawa_family(),
        "shimura_period": b.bench_shimura_period_family(),
        "arithmetic_arnold": b.bench_arithmetic_arnold_family(),
        "darmon_point": b.bench_darmon_point_family(),
        "bertolini_darmon": b.bench_bertolini_darmon_family(),
        "howard_main": b.bench_howard_main_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
