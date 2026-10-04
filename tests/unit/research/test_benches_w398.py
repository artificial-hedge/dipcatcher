"""Wave-398 representation-theory-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w398 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "induced_char": b.bench_induced_char_family(),
        "artins_theorem": b.bench_artins_theorem_family(),
        "tensor_char": b.bench_tensor_char_family(),
        "clifford_toy": b.bench_clifford_toy_family(),
        "schur_index": b.bench_schur_index_family(),
        "frobenius_group": b.bench_frobenius_group_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
