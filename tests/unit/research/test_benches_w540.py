"""Wave-540 transcendence-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w540 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hermite_lindemann": b.bench_hermite_lindemann_family(),
        "gelfond_schneider": b.bench_gelfond_schneider_family(),
        "baker_thm": b.bench_baker_thm_family(),
        "lindemann_weier": b.bench_lindemann_weier_family(),
        "schanuel_conj": b.bench_schanuel_conj_family(),
        "siegel_shidlovskii": b.bench_siegel_shidlovskii_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
