"""Wave-866 model-order-reduction adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w866 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "pod_galerkin": b.bench_pod_galerkin_family(),
        "reduced_basis": b.bench_reduced_basis_family(),
        "deim_point": b.bench_deim_point_family(),
        "greedy_rb": b.bench_greedy_rb_family(),
        "eim_interp": b.bench_eim_interp_family(),
        "proper_gen": b.bench_proper_gen_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
