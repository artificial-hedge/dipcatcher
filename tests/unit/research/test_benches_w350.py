"""Wave-350 representation-theory adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w350 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "character_table_s3": b.bench_character_table_s3_family(),
        "perm_rep": b.bench_perm_rep_family(),
        "schur_ortho": b.bench_schur_ortho_family(),
        "induced_rep": b.bench_induced_rep_family(),
        "fourier_sn": b.bench_fourier_sn_family(),
        "regular_rep": b.bench_regular_rep_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
