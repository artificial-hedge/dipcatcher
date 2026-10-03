"""Wave-724 arithmetic-cycles adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w724 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "coates_wiles": b.bench_coates_wiles_family(),
        "iwasawa_lfunc": b.bench_iwasawa_lfunc_family(),
        "greenberg_selmer": b.bench_greenberg_selmer_family(),
        "kurihara_iwasawa": b.bench_kurihara_iwasawa_family(),
        "heegner_cycle": b.bench_heegner_cycle_family(),
        "gan_gross_prasad": b.bench_gan_gross_prasad_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
