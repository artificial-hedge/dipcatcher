"""Wave-894 interpolation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w894 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "lagrange_interp": b.bench_lagrange_interp_family(),
        "neville_interp": b.bench_neville_interp_family(),
        "hermite_interp": b.bench_hermite_interp_family(),
        "divid_diff_table": b.bench_divid_diff_table_family(),
        "barycentric_wts": b.bench_barycentric_wts_family(),
        "floater_hormann": b.bench_floater_hormann_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
