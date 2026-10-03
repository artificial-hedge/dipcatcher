"""Wave-362 complex-analysis adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w362 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cauchy_integral": b.bench_cauchy_integral_family(),
        "residue_calc": b.bench_residue_calc_family(),
        "laurent_series": b.bench_laurent_series_family(),
        "argument_principle": b.bench_argument_principle_family(),
        "conformal_map": b.bench_conformal_map_family(),
        "liouville": b.bench_liouville_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
