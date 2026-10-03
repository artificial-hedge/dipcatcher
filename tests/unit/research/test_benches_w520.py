"""Wave-520 automorphic-GL(n) adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w520 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gln_automorphic": b.bench_gln_automorphic_family(),
        "whittaker_model": b.bench_whittaker_model_family(),
        "godement_jacq": b.bench_godement_jacq_family(),
        "rankin_selberg": b.bench_rankin_selberg_family(),
        "langlands_lfunc": b.bench_langlands_lfunc_family(),
        "converse_thm": b.bench_converse_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
