"""Wave-53 bench-adapter wiring tests (benches_w53.py + catalog wiring)."""

from __future__ import annotations

import math

import pytest

from quant_fund.research import benches_w53
from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS

WAVE53_FAMILIES = (
    "corradi_swanson",
    "engle_kroner_bekk",
    "heston_qe",
    "model_confidence_set",
    "christoffersen_pelletier",
    "sheppard_heavy",
)


def _adapter(name: str):
    return getattr(benches_w53, f"bench_{name}")


def test_all_wave53_families_registered():
    from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

    for name in WAVE53_FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES


def test_adapters_exist_and_emit_finite_blobs():
    for name in WAVE53_FAMILIES:
        out = _adapter(name)()
        assert isinstance(out, dict), name
        assert out, f"{name} emitted empty blob"
        for k, v in out.items():
            assert isinstance(v, float), (name, k)
            assert math.isfinite(v), (name, k)


def test_metric_keys_avoid_forbidden_tokens():
    for name in WAVE53_FAMILIES:
        for k in _adapter(name)():
            toks = set(k.lower().split("_"))
            assert not (toks & FORBIDDEN_RESEARCH_METRIC_KEYS), (name, k)


def test_adapters_are_deterministic():
    for name in WAVE53_FAMILIES:
        a = _adapter(name)()
        b = _adapter(name)()
        assert a == b, name


def test_missing_module_degrades_to_empty(monkeypatch: pytest.MonkeyPatch):
    import sys

    monkeypatch.setitem(sys.modules, "quant_fund.models.corradi_swanson", None)
    assert benches_w53.bench_corradi_swanson() == {}


def test_nonfinite_blob_filtered(monkeypatch: pytest.MonkeyPatch):
    class _Fake:
        @staticmethod
        def bench_corradi_swanson(seed: int = 0) -> dict[str, float]:
            return {"synthetic_a": 1.0, "synthetic_nan": float("nan")}

    monkeypatch.setattr(
        "quant_fund.models.corradi_swanson.bench_corradi_swanson",
        _Fake.bench_corradi_swanson,
    )
    assert benches_w53.bench_corradi_swanson() == {}
