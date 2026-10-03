"""Wave-413 Galois-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w413 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "artin_lemma": b.bench_artin_lemma_family(),
        "normal_basis": b.bench_normal_basis_family(),
        "kummer_ext": b.bench_kummer_ext_family(),
        "abelian_ext": b.bench_abelian_ext_family(),
        "frobenius_el": b.bench_frobenius_el_family(),
        "inseparable": b.bench_inseparable_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
