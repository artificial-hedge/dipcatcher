"""Wave-89 bench-adapter wiring tests (benches_w89.py + catalog wiring)."""

from __future__ import annotations

import math

from quant_fund.research import benches_w89
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

WAVE89_FAMILIES = (
    "edf_tests",
    "normality_tests",
    "scale_homogeneity",
    "score_scale",
    "het_regressions",
    "serial_diagnostics",
)


def _adapter(name: str):
    return getattr(benches_w89, f"bench_{name}")


def test_all_wave89_families_registered():
    from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

    for name in WAVE89_FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES


def test_adapters_exist_and_emit_finite_blobs():
    for name in WAVE89_FAMILIES:
        out = _adapter(name)()
        assert isinstance(out, dict), name
        assert out, f"{name} emitted empty blob"
        for k, v in out.items():
            assert isinstance(v, float), (name, k)
            assert math.isfinite(v), (name, k)


def test_no_forbidden_headline_tokens():
    for name in WAVE89_FAMILIES:
        for key in _adapter(name)():
            assert key not in FORBIDDEN_RESEARCH_METRIC_KEYS
