"""Wave-365 probability-3 adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w365 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "optional_stopping": b.bench_optional_stopping_family(),
        "doob_decomp": b.bench_doob_decomp_family(),
        "martingale_clt": b.bench_martingale_clt_family(),
        "azuma": b.bench_azuma_family(),
        "coupling_arg": b.bench_coupling_arg_family(),
        "ergodic_thm": b.bench_ergodic_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
