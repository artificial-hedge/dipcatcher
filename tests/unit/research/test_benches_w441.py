"""Wave-441 Galois-representations adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w441 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gal_rep": b.bench_gal_rep_family(),
        "fontaine_ring": b.bench_fontaine_ring_family(),
        "filtered_module": b.bench_filtered_module_family(),
        "weil_deligne": b.bench_weil_deligne_family(),
        "hecke_eigensys": b.bench_hecke_eigensys_family(),
        "ribet_toy": b.bench_ribet_toy_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
