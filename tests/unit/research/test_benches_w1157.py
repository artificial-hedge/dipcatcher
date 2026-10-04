"""Wave-1157 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1157 import (
    bench_artificial_intelligence_family,
    bench_computer_science_2_family,
    bench_data_engineering_family,
    bench_information_theory_2_family,
    bench_machine_learning_2_family,
    bench_software_engineering_family,
)

_FAMILY_BENCHES = [
    bench_computer_science_2_family,
    bench_software_engineering_family,
    bench_machine_learning_2_family,
    bench_artificial_intelligence_family,
    bench_data_engineering_family,
    bench_information_theory_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
