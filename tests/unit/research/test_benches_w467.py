"""Wave-467 set-theory-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w467 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "constructible_l": b.bench_constructible_l_family(),
        "large_card": b.bench_large_card_family(),
        "pcf_theory": b.bench_pcf_theory_family(),
        "proper_forcing": b.bench_proper_forcing_family(),
        "core_model": b.bench_core_model_family(),
        "square_princ": b.bench_square_princ_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
