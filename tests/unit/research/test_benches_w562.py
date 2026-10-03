"""Wave-562 symplectic-geometry-2 adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w562 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "gromov_width": b.bench_gromov_width_family(),
        "hofer_metric": b.bench_hofer_metric_family(),
        "symplectic_capacity": b.bench_symplectic_capacity_family(),
        "symplectic_packing": b.bench_symplectic_packing_family(),
        "mcduff_polterovich": b.bench_mcduff_polterovich_family(),
        "ekeland_hofer": b.bench_ekeland_hofer_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
