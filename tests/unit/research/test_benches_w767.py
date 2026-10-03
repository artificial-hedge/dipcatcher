"""Wave-767 Gaussian-process adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w767 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "slepian_lemma": b.bench_slepian_lemma_family(),
        "fernique_thm": b.bench_fernique_thm_family(),
        "borell_tis": b.bench_borell_tis_family(),
        "sudakov_min": b.bench_sudakov_min_family(),
        "talagrand_conc": b.bench_talagrand_conc_family(),
        "gordon_thm": b.bench_gordon_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
