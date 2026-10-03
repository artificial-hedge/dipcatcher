"""Wave-552 contact-topology adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w552 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "contact_form": b.bench_contact_form_family(),
        "legendrian_knot": b.bench_legendrian_knot_family(),
        "overtwisted": b.bench_overtwisted_family(),
        "tight_contact": b.bench_tight_contact_family(),
        "giroux_corr": b.bench_giroux_corr_family(),
        "convex_surface": b.bench_convex_surface_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
