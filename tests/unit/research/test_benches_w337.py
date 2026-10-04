"""Wave-337 lambda-calculus/rewriting adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w337 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "ski_combinator": b.bench_ski_combinator_family(),
        "de_bruijn": b.bench_de_bruijn_family(),
        "church_encoding": b.bench_church_encoding_family(),
        "lambda_typing": b.bench_lambda_typing_family(),
        "unification": b.bench_unification_family(),
        "knuth_bendix": b.bench_knuth_bendix_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
