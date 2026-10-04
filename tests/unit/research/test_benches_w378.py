"""Wave-378 forcing/set-theory-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w378 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "forcing_poset": b.bench_forcing_poset_family(),
        "dense_filter": b.bench_dense_filter_family(),
        "names_eval": b.bench_names_eval_family(),
        "cohen_adds": b.bench_cohen_adds_family(),
        "ma_toy": b.bench_ma_toy_family(),
        "large_cardinal": b.bench_large_cardinal_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
