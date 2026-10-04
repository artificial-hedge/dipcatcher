"""Wave-526 complex-dynamics adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w526 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "julia_set": b.bench_julia_set_family(),
        "mandelbrot_set": b.bench_mandelbrot_set_family(),
        "fatou_set": b.bench_fatou_set_family(),
        "sullivan_no_wander": b.bench_sullivan_no_wander_family(),
        "douady_hubbard": b.bench_douady_hubbard_family(),
        "parabolic_impl": b.bench_parabolic_impl_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
