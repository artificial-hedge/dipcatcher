"""Wave-1007 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1007 import (
    bench_bose_einstein_family,
    bench_fermi_dirac_family,
    bench_free_energy_family,
    bench_gibbs_measure_family,
    bench_ising_model_family,
    bench_partition_function_family,
)

_FAMILY_BENCHES = [
    bench_ising_model_family,
    bench_partition_function_family,
    bench_bose_einstein_family,
    bench_fermi_dirac_family,
    bench_gibbs_measure_family,
    bench_free_energy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
