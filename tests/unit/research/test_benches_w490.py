"""Wave-490 arithmetic-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w490 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "arakelov_deg": b.bench_arakelov_deg_family(),
        "adelic_curve": b.bench_adelic_curve_family(),
        "height_arakelov": b.bench_height_arakelov_family(),
        "faltings_metric": b.bench_faltings_metric_family(),
        "arithmetic_chow": b.bench_arithmetic_chow_family(),
        "arith_rr": b.bench_arith_rr_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
