"""Wave-480 infinity-topos-3 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w480 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "etale_geom": b.bench_etale_geom_family(),
        "gros_topos": b.bench_gros_topos_family(),
        "local_homeo": b.bench_local_homeo_family(),
        "classify_obj": b.bench_classify_obj_family(),
        "pi_infty": b.bench_pi_infty_family(),
        "exponentiable": b.bench_exponentiable_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
