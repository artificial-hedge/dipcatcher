"""Wave-1186 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1186 import (
    bench_accessibility_studies_family,
    bench_hci_studies_family,
    bench_information_architecture_family,
    bench_interaction_design_family,
    bench_service_design_family,
    bench_ux_design_family,
)

_FAMILY_BENCHES = [
    bench_ux_design_family,
    bench_hci_studies_family,
    bench_information_architecture_family,
    bench_interaction_design_family,
    bench_accessibility_studies_family,
    bench_service_design_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
