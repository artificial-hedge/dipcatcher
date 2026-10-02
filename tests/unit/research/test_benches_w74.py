"""Wave-74 bench-adapter wiring tests (benches_w74.py + catalog wiring)."""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w74
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

WAVE74_FAMILIES = (
    "dispersion_tests",
    "median_tests",
    "cochran_q",
    "quade",
    "van_der_waerden",
    "dunn_test",
)


def _adapter(name: str):
    return getattr(benches_w74, f"bench_{name}")


def test_all_wave74_families_registered():
    from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

    for name in WAVE74_FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES


def test_adapters_exist_and_emit_finite_blobs():
    for name in WAVE74_FAMILIES:
        out = _adapter(name)()
        assert isinstance(out, dict), name
        assert out, f"{name} emitted empty blob"
        for k, v in out.items():
            assert isinstance(v, float), (name, k)
            assert math.isfinite(v), (name, k)


def test_adapters_emit_no_forbidden_metrics():
    for name in WAVE74_FAMILIES:
        for k in _adapter(name)():
            toks = set(k.lower().split("_"))
            assert not (toks & FORBIDDEN_RESEARCH_METRIC_KEYS), (name, k)


def test_adapters_are_deterministic():
    for name in WAVE74_FAMILIES:
        a = _adapter(name)()
        b = _adapter(name)()
        assert a == b, name


def test_missing_module_degrades_to_empty(monkeypatch: pytest.MonkeyPatch):
    import sys

    monkeypatch.setitem(sys.modules, "quant_fund.models.dispersion_tests", None)
    assert benches_w74.bench_dispersion_tests() == {}


def test_nonfinite_blob_filtered(monkeypatch: pytest.MonkeyPatch):
    class _Fake:
        @staticmethod
        def bench_dispersion_tests(seed: int = 0) -> dict[str, float]:
            return {"synthetic_a": 1.0, "synthetic_nan": float("nan")}

    monkeypatch.setattr(
        "quant_fund.models.dispersion_tests.bench_dispersion_tests",
        _Fake.bench_dispersion_tests,
    )
    assert benches_w74.bench_dispersion_tests() == {}
