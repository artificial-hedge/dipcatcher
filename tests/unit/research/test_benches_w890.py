"""Wave-890 special-function adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w890 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "zeta_fn": b.bench_zeta_fn_family(),
        "elliptic_fn": b.bench_elliptic_fn_family(),
        "hartley_transform": b.bench_hartley_transform_family(),
        "radon_transform": b.bench_radon_transform_family(),
        "clenshaw_quad": b.bench_clenshaw_quad_family(),
        "fejer_nested": b.bench_fejer_nested_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
