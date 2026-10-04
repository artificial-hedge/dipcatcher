"""Wave-430 higher-algebra adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w430 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "e_n_algebra": b.bench_e_n_algebra_family(),
        "operad_infty": b.bench_operad_infty_family(),
        "monoidal_infty": b.bench_monoidal_infty_family(),
        "module_cat": b.bench_module_cat_family(),
        "brane_tensor": b.bench_brane_tensor_family(),
        "delooping": b.bench_delooping_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
