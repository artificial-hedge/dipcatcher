"""Wave-573 Hodge-2/periods adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w573 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "griffiths_transv": b.bench_griffiths_transv_family(),
        "period_domain": b.bench_period_domain_family(),
        "mumford_tate": b.bench_mumford_tate_family(),
        "hodge_class": b.bench_hodge_class_family(),
        "absolute_hodge": b.bench_absolute_hodge_family(),
        "hodge_conj": b.bench_hodge_conj_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
