"""Wave-581 spectral-sequences adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w581 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "serre_ss4": b.bench_serre_ss4_family(),
        "bockstein_ss": b.bench_bockstein_ss_family(),
        "eilenberg_moore": b.bench_eilenberg_moore_family(),
        "bousfield_ss": b.bench_bousfield_ss_family(),
        "lyndon_ss": b.bench_lyndon_ss_family(),
        "cartan_ss": b.bench_cartan_ss_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
