"""Wave-435 derived-schemes adapter tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w435 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "derived_scheme": b.bench_derived_scheme_family(),
        "quasi_coherent": b.bench_quasi_coherent_family(),
        "derived_fiber": b.bench_derived_fiber_family(),
        "spectral_scheme": b.bench_spectral_scheme_family(),
        "virtual_class": b.bench_virtual_class_family(),
        "shifted_symplectic": b.bench_shifted_symplectic_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
