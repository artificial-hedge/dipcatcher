"""Wave-587 birational-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w587 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "terminal_sing": b.bench_terminal_sing_family(),
        "canonical_sing2": b.bench_canonical_sing2_family(),
        "klt_mmp": b.bench_klt_mmp_family(),
        "mmp_flip": b.bench_mmp_flip_family(),
        "abundance_conj": b.bench_abundance_conj_family(),
        "bdd_fano": b.bench_bdd_fano_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
