"""Wave-519 modular-forms adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w519 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "modular_form": b.bench_modular_form_family(),
        "hecke_op2": b.bench_hecke_op2_family(),
        "eisenstein_srs2": b.bench_eisenstein_srs2_family(),
        "cusp_form": b.bench_cusp_form_family(),
        "theta_func": b.bench_theta_func_family(),
        "dedekind_eta": b.bench_dedekind_eta_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
