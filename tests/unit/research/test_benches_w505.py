"""Wave-505 cobordism-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w505 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "cobordism_grp": b.bench_cobordism_grp_family(),
        "oriented_cob": b.bench_oriented_cob_family(),
        "unoriented_cob": b.bench_unoriented_cob_family(),
        "complex_cob": b.bench_complex_cob_family(),
        "framed_cob": b.bench_framed_cob_family(),
        "thom_cob": b.bench_thom_cob_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
