"""Wave-399 model-theory-5 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w399 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ef_game_toy": b.bench_ef_game_toy_family(),
        "vaught_test": b.bench_vaught_test_family(),
        "real_closed": b.bench_real_closed_family(),
        "boolean_prime": b.bench_boolean_prime_family(),
        "fraisse_limit": b.bench_fraisse_limit_family(),
        "qe_dense_order": b.bench_qe_dense_order_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
