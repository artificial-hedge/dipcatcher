"""Wave-500 arithmetic-Langlands adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w500 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "local_langlands": b.bench_local_langlands_family(),
        "harris_taylor": b.bench_harris_taylor_family(),
        "weil_group": b.bench_weil_group_family(),
        "langlands_functoriality": b.bench_langlands_functoriality_family(),
        "epsilon_factor": b.bench_epsilon_factor_family(),
        "l_packet": b.bench_l_packet_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
