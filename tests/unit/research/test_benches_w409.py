"""Wave-409 homotopy-5 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w409 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "spectral_seq2": b.bench_spectral_seq2_family(),
        "eilenberg_zilber": b.bench_eilenberg_zilber_family(),
        "dold_kan": b.bench_dold_kan_family(),
        "postnikov": b.bench_postnikov_family(),
        "stable_range": b.bench_stable_range_family(),
        "cohend": b.bench_cohend_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
