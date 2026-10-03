"""Wave-760 Brownian-motion adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w760 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "levy_bm": b.bench_levy_bm_family(),
        "wiener_bm": b.bench_wiener_bm_family(),
        "doob_bm": b.bench_doob_bm_family(),
        "ito_bm": b.bench_ito_bm_family(),
        "cameron_martin": b.bench_cameron_martin_family(),
        "gikhman_skorokhod": b.bench_gikhman_skorokhod_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
