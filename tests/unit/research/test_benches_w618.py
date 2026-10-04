"""Wave-618 p-adic-5 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w618 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fontaine_curve": b.bench_fontaine_curve_family(),
        "untilt": b.bench_untilt_family(),
        "perfectoid_c": b.bench_perfectoid_c_family(),
        "b_drb": b.bench_b_drb_family(),
        "phi_mod": b.bench_phi_mod_family(),
        "ad_period": b.bench_ad_period_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
