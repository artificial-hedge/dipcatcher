"""Wave-450 stable-infinity adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w450 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "stable_infty": b.bench_stable_infty_family(),
        "spectra_cat": b.bench_spectra_cat_family(),
        "exact_seq": b.bench_exact_seq_family(),
        "stable_tstruct": b.bench_stable_tstruct_family(),
        "smash_monoidal": b.bench_smash_monoidal_family(),
        "thh_tc": b.bench_thh_tc_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
