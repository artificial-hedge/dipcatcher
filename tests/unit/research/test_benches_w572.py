"""Wave-572 abelian-varieties adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w572 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "abelian_variety": b.bench_abelian_variety_family(),
        "isogeny_av": b.bench_isogeny_av_family(),
        "tate_module": b.bench_tate_module_family(),
        "shafarevich_conj": b.bench_shafarevich_conj_family(),
        "faltings_thm": b.bench_faltings_thm_family(),
        "mordell_weil_av": b.bench_mordell_weil_av_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
