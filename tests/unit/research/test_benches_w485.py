"""Wave-485 p-adic-4 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w485 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "lubin_tate2": b.bench_lubin_tate2_family(),
        "bc_space": b.bench_bc_space_family(),
        "local_shimura": b.bench_local_shimura_family(),
        "scholze_weinstein": b.bench_scholze_weinstein_family(),
        "fargues_curve2": b.bench_fargues_curve2_family(),
        "banach_colmez2": b.bench_banach_colmez2_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
