"""Wave-803 signature adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w803 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "signature_kernel": b.bench_signature_kernel_family(),
        "pde_signature": b.bench_pde_signature_family(),
        "truncated_sig": b.bench_truncated_sig_family(),
        "signature_gan2": b.bench_signature_gan2_family(),
        "expected_sig": b.bench_expected_sig_family(),
        "sig_inversion": b.bench_sig_inversion_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
