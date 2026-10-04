"""Wave-343 lattice/universal-algebra adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w343 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "lattice_check": b.bench_lattice_check_family(),
        "galois_connection": b.bench_galois_connection_family(),
        "tarski_fixed": b.bench_tarski_fixed_family(),
        "boolean_algebra": b.bench_boolean_algebra_family(),
        "congruence_lattice": b.bench_congruence_lattice_family(),
        "term_algebra": b.bench_term_algebra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
