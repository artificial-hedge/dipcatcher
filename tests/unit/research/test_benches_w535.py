"""Wave-535 microlocal-analysis adapter tests (SYNTHETIC)."""

from __future__ import annotations

import quant_fund.research.benches_w535 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "wavefront_set": b.bench_wavefront_set_family(),
        "pseudodiff_op": b.bench_pseudodiff_op_family(),
        "fourier_io": b.bench_fourier_io_family(),
        "symbol_calc": b.bench_symbol_calc_family(),
        "propagation_sing": b.bench_propagation_sing_family(),
        "elliptic_est": b.bench_elliptic_est_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
