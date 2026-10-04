"""Wave-376 homotopy-theory-2 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w376 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "fibration": b.bench_fibration_family(),
        "cofibration": b.bench_cofibration_family(),
        "serre_ss": b.bench_serre_ss_family(),
        "whitehead": b.bench_whitehead_family(),
        "suspension": b.bench_suspension_family(),
        "spectra": b.bench_spectra_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
