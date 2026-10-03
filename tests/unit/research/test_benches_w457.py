"""Wave-457 double-category/proarrow adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w457 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "proarrow": b.bench_proarrow_family(),
        "virtual_equip": b.bench_virtual_equip_family(),
        "fibrant_double": b.bench_fibrant_double_family(),
        "tabulation": b.bench_tabulation_family(),
        "companion_conj": b.bench_companion_conj_family(),
        "framed_bicat": b.bench_framed_bicat_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
