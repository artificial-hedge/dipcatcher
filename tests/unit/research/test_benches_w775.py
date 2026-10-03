"""Wave-775 martingale-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w775 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "doleans_meas": b.bench_doleans_meas_family(),
        "predictable_proc": b.bench_predictable_proc_family(),
        "local_mart": b.bench_local_mart_family(),
        "square_bracket": b.bench_square_bracket_family(),
        "burkholder_davis": b.bench_burkholder_davis_family(),
        "gundy_mart": b.bench_gundy_mart_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
