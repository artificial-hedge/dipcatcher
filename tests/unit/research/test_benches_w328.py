"""Wave-328 bench adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w328 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "path_types": b.bench_path_types_family(),
        "hlevel_check": b.bench_hlevel_check_family(),
        "univalence_toy": b.bench_univalence_toy_family(),
        "kan_hcomp": b.bench_kan_hcomp_family(),
        "funext_toy": b.bench_funext_toy_family(),
        "hit_quotient": b.bench_hit_quotient_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
