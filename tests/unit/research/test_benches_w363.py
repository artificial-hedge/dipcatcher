"""Wave-363 real-analysis adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w363 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cantor_set": b.bench_cantor_set_family(),
        "baire_category": b.bench_baire_category_family(),
        "vitali_set": b.bench_vitali_set_family(),
        "egorov_thm": b.bench_egorov_thm_family(),
        "fatou_lemma": b.bench_fatou_lemma_family(),
        "monotone_conv": b.bench_monotone_conv_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
