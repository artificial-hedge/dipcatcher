"""Wave-840 rational-approximation adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w840 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "pade_approx": b.bench_pade_approx_family(),
        "rational_chebyshev": b.bench_rational_chebyshev_family(),
        "stieltjes_fraction": b.bench_stieltjes_fraction_family(),
        "loewner_interp": b.bench_loewner_interp_family(),
        "nevanlinna_pick": b.bench_nevanlinna_pick_family(),
        "schur_continued": b.bench_schur_continued_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
