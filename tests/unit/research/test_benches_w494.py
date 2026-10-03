"""Wave-494 noncommutative-geometry adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w494 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "hochschild_coh": b.bench_hochschild_coh_family(),
        "cyclic_coh": b.bench_cyclic_coh_family(),
        "nc_scheme": b.bench_nc_scheme_family(),
        "calabi_yau_alg": b.bench_calabi_yau_alg_family(),
        "ginzburg_dga": b.bench_ginzburg_dga_family(),
        "connes_nc": b.bench_connes_nc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
