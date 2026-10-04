"""Wave-762 weak-convergence adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w762 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "martin_boundary": b.bench_martin_boundary_family(),
        "doob_meyer": b.bench_doob_meyer_family(),
        "cadlag_space": b.bench_cadlag_space_family(),
        "skohorod_metric": b.bench_skohorod_metric_family(),
        "prohorov_thm2": b.bench_prohorov_thm2_family(),
        "tightness_check": b.bench_tightness_check_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
