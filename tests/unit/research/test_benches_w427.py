"""Wave-427 number-theory-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w427 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "elliptic_height": b.bench_elliptic_height_family(),
        "mordell_weil": b.bench_mordell_weil_family(),
        "lseries_toy": b.bench_lseries_toy_family(),
        "bsd_toy": b.bench_bsd_toy_family(),
        "modularity_toy": b.bench_modularity_toy_family(),
        "padic_integral": b.bench_padic_integral_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
