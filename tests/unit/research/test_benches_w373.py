"""Wave-373 optimization-3 adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w373 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bundle_method": b.bench_bundle_method_family(),
        "sqp": b.bench_sqp_family(),
        "ip_qp": b.bench_ip_qp_family(),
        "trust_region": b.bench_trust_region_family(),
        "frank_wolfe2": b.bench_frank_wolfe2_family(),
        "bfgs_wolfe": b.bench_bfgs_wolfe_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
