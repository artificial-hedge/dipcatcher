"""Wave-340 homological-algebra adapter smoke tests (SYNTHETIC keys + ranges)."""

from __future__ import annotations

import quant_fund.research.benches_w340 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "chain_complex": b.bench_chain_complex_family(),
        "tor_ext": b.bench_tor_ext_family(),
        "sheaf_check": b.bench_sheaf_check_family(),
        "hilbert_series": b.bench_hilbert_series_family(),
        "snake_lemma": b.bench_snake_lemma_family(),
        "variety_morph": b.bench_variety_morph_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
