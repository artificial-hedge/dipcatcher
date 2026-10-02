"""Wave-91 bench-adapter wiring tests (benches_w91.py + catalog wiring)."""

from __future__ import annotations

import math

from quant_fund.research import benches_w91
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

WAVE91_FAMILIES = (
    "unconstrained_optimizers",
    "clustering_methods",
    "manifold_learning",
    "robust_regression",
    "empirical_bayes",
    "design_experiments",
)


def _adapter(name: str):
    return getattr(benches_w91, f"bench_{name}")


def test_all_wave91_families_registered():
    from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

    for name in WAVE91_FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES


def test_adapters_exist_and_emit_finite_blobs():
    for name in WAVE91_FAMILIES:
        out = _adapter(name)()
        assert isinstance(out, dict), name
        assert out, f"{name} emitted empty blob"
        for k, v in out.items():
            assert isinstance(v, float), (name, k)
            assert math.isfinite(v), (name, k)


def test_no_forbidden_headline_tokens():
    for name in WAVE91_FAMILIES:
        for key in _adapter(name)():
            assert key not in FORBIDDEN_RESEARCH_METRIC_KEYS
