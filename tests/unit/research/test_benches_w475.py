"""Wave-475 homotopical-algebra adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w475 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "dendroidal": b.bench_dendroidal_family(),
        "infty_operad": b.bench_infty_operad_family(),
        "cyclic_hk": b.bench_cyclic_hk_family(),
        "chiral_alg": b.bench_chiral_alg_family(),
        "sifted_cat": b.bench_sifted_cat_family(),
        "seq_spectra": b.bench_seq_spectra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
