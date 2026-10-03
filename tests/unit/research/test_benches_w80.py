"""Wave-80 bench-adapter wiring tests (benches_w80.py + catalog wiring)."""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w80
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

WAVE80_FAMILIES = (
    "rmst",
    "ois_curve",
    "henze_zirkler",
    "epps_singleton",
    "watson",
    "energy_test",
)


def _adapter(name: str):
    return getattr(benches_w80, f"bench_{name}")


def test_all_wave80_families_registered():
    from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

    for name in WAVE80_FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES


def test_adapters_exist_and_emit_finite_blobs():
    for name in WAVE80_FAMILIES:
        out = _adapter(name)()
        assert isinstance(out, dict), name
        assert out, f"{name} emitted empty blob"
        for k, v in out.items():
            assert isinstance(v, float), (name, k)
            assert math.isfinite(v), (name, k)


def test_adapters_emit_no_forbidden_metrics():
    for name in WAVE80_FAMILIES:
        for k in _adapter(name)():
            toks = set(k.lower().split("_"))
            assert not (toks & FORBIDDEN_RESEARCH_METRIC_KEYS), (name, k)


def test_adapters_are_deterministic():
    for name in WAVE80_FAMILIES:
        a = _adapter(name)()
        b = _adapter(name)()
        assert a == b, name


def test_missing_module_degrades_to_empty(monkeypatch: pytest.MonkeyPatch):
    import sys

    monkeypatch.setitem(sys.modules, "quant_fund.models.rmst", None)
    assert benches_w80.bench_rmst() == {}


def test_nonfinite_blob_filtered(monkeypatch: pytest.MonkeyPatch):
    class _Fake:
        @staticmethod
        def bench_rmst(seed: int = 0) -> dict[str, float]:
            return {"synthetic_a": 1.0, "synthetic_nan": float("nan")}

    monkeypatch.setattr(
        "quant_fund.models.rmst.bench_rmst",
        _Fake.bench_rmst,
    )
    assert benches_w80.bench_rmst() == {}
