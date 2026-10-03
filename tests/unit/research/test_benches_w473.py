"""Wave-473 higher-algebra-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w473 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "thh_2": b.bench_thh_2_family(),
        "cyclotomic2": b.bench_cyclotomic2_family(),
        "cartier_mod": b.bench_cartier_mod_family(),
        "witt_vec2": b.bench_witt_vec2_family(),
        "crystalline_stack": b.bench_crystalline_stack_family(),
        "trt_functor": b.bench_trt_functor_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
