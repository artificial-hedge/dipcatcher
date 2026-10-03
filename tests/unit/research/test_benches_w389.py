"""Wave-389 number-fields adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w389 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "norm_subring": b.bench_norm_subring_family(),
        "discriminant_field": b.bench_discriminant_field_family(),
        "decomposition_group": b.bench_decomposition_group_family(),
        "ramification": b.bench_ramification_family(),
        "artin_symbol": b.bench_artin_symbol_family(),
        "class_group_toy": b.bench_class_group_toy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
