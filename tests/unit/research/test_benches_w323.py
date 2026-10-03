"""Wave-323 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w323 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "free_monad": b.bench_free_monad_family(),
        "alg_effects": b.bench_alg_effects_family(),
        "shift_reset": b.bench_shift_reset_family(),
        "row_types": b.bench_row_types_family(),
        "session_types": b.bench_session_types_family(),
        "gradual_types": b.bench_gradual_types_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
