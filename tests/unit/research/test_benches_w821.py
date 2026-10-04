"""Wave-821 random-measure adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w821 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "random_measure": b.bench_random_measure_family(),
        "integer_measure": b.bench_integer_measure_family(),
        "poisson_rm": b.bench_poisson_rm_family(),
        "compensator_rm": b.bench_compensator_rm_family(),
        "jump_measure": b.bench_jump_measure_family(),
        "sato_measure": b.bench_sato_measure_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
