"""Wave-749 vertex-model-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w749 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "bufetov_sixv": b.bench_bufetov_sixv_family(),
        "borodin_bufetov": b.bench_borodin_bufetov_family(),
        "kuan_sixv": b.bench_kuan_sixv_family(),
        "dimitrov_sixv": b.bench_dimitrov_sixv_family(),
        "borodin_wheeler": b.bench_borodin_wheeler_family(),
        "wheeler_zinn": b.bench_wheeler_zinn_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
