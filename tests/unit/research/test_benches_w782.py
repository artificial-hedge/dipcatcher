"""Wave-782 filtration/Jacod-Shiryaev adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w782 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "pinsky_proc": b.bench_pinsky_proc_family(),
        "ffusion_lims": b.bench_ffusion_lims_family(),
        "kunita_watanabe": b.bench_kunita_watanabe_family(),
        "filt_proc": b.bench_filt_proc_family(),
        "slivnyak": b.bench_slivnyak_family(),
        "jacod_shiryaev": b.bench_jacod_shiryaev_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
