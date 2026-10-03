"""Wave-468 model-theory-7 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w468 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "o_minimal": b.bench_o_minimal_family(),
        "nip_theory": b.bench_nip_theory_family(),
        "nonforking": b.bench_nonforking_family(),
        "simple_theory": b.bench_simple_theory_family(),
        "abstract_erc": b.bench_abstract_erc_family(),
        "tame_metric": b.bench_tame_metric_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
