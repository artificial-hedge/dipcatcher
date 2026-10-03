"""Wave-421 category-theory-4 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w421 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "traced_monoidal": b.bench_traced_monoidal_family(),
        "star_autonomous": b.bench_star_autonomous_family(),
        "frobenius_alg": b.bench_frobenius_alg_family(),
        "span_compose": b.bench_span_compose_family(),
        "profunctor_toy": b.bench_profunctor_toy_family(),
        "endo_coend": b.bench_endo_coend_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
