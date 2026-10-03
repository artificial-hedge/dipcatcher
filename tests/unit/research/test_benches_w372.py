"""Wave-372 stochastic-analysis adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w372 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ito_lemma": b.bench_ito_lemma_family(),
        "girsanov": b.bench_girsanov_family(),
        "sde_strong": b.bench_sde_strong_family(),
        "local_time": b.bench_local_time_family(),
        "quadratic_var": b.bench_quadratic_var_family(),
        "malliavin": b.bench_malliavin_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
