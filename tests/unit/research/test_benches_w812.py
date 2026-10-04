"""Wave-812 regenerative adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w812 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "regenerative": b.bench_regenerative_family(),
        "epsilon_coupling": b.bench_epsilon_coupling_family(),
        "small_set": b.bench_small_set_family(),
        "petite_set": b.bench_petite_set_family(),
        "split_chain": b.bench_split_chain_family(),
        "nummelin": b.bench_nummelin_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
