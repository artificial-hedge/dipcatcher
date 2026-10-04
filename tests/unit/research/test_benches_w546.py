"""Wave-546 knot-theory adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w546 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "knot_invariant": b.bench_knot_invariant_family(),
        "jones_poly": b.bench_jones_poly_family(),
        "alexander_poly": b.bench_alexander_poly_family(),
        "knot_group": b.bench_knot_group_family(),
        "knot_signature": b.bench_knot_signature_family(),
        "vassiliev_inv": b.bench_vassiliev_inv_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
