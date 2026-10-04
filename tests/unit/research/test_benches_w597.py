"""Wave-597 chromatic-homotopy adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w597 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "curtis_lower": b.bench_curtis_lower_family(),
        "bousfield_kan": b.bench_bousfield_kan_family(),
        "lannes_t": b.bench_lannes_t_family(),
        "dror_smith": b.bench_dror_smith_family(),
        "telescope_conj": b.bench_telescope_conj_family(),
        "periodicity_thm": b.bench_periodicity_thm_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
