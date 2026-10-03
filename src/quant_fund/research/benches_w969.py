"""Wave-969 operator K-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bott_periodicity_k import bench_bott_periodicity_k
from quant_fund.models.elliott_invariant import bench_elliott_invariant
from quant_fund.models.k0_algebra import bench_k0_algebra
from quant_fund.models.k1_algebra import bench_k1_algebra
from quant_fund.models.pimsner_voicul import bench_pimsner_voicul
from quant_fund.models.six_term_exact import bench_six_term_exact

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_k0_algebra_family(seed: int = _SEED + 35700) -> dict[str, float]:
    return _finite_blob(bench_k0_algebra(seed))


def bench_k1_algebra_family(seed: int = _SEED + 35701) -> dict[str, float]:
    return _finite_blob(bench_k1_algebra(seed))


def bench_bott_periodicity_k_family(seed: int = _SEED + 35702) -> dict[str, float]:
    return _finite_blob(bench_bott_periodicity_k(seed))


def bench_six_term_exact_family(seed: int = _SEED + 35703) -> dict[str, float]:
    return _finite_blob(bench_six_term_exact(seed))


def bench_pimsner_voicul_family(seed: int = _SEED + 35704) -> dict[str, float]:
    return _finite_blob(bench_pimsner_voicul(seed))


def bench_elliott_invariant_family(seed: int = _SEED + 35705) -> dict[str, float]:
    return _finite_blob(bench_elliott_invariant(seed))
