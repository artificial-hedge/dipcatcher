"""Wave-331 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w331 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "np_reduce": b.bench_np_reduce_family(),
        "fpras_dnf": b.bench_fpras_dnf_family(),
        "sumcheck": b.bench_sumcheck_family(),
        "param_fpt": b.bench_param_fpt_family(),
        "pcp_verify": b.bench_pcp_verify_family(),
        "circuit_lb": b.bench_circuit_lb_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
