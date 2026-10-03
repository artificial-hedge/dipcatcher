"""Wave-394 order-theory adapter smoke tests (SYNTHETIC keys)."""

from __future__ import annotations

import quant_fund.research.benches_w394 as b


def test_families_emit_synthetic_scores() -> None:
    outs = {
        "downset_lattice": b.bench_downset_lattice_family(),
        "zeta_mobius": b.bench_zeta_mobius_family(),
        "linear_extension": b.bench_linear_extension_family(),
        "sperner_bound": b.bench_sperner_bound_family(),
        "dilworth_partition": b.bench_dilworth_partition_family(),
        "birkhoff_rep": b.bench_birkhoff_rep_family(),
    }
    for name, out in outs.items():
        key = f"synthetic_{name}"
        assert key in out, f"{key} missing"
        assert 0.0 <= out[key] <= 1.0, f"{key}={out[key]} out of range"
