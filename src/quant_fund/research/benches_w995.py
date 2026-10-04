"""Wave-995 integrable-systems canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.calogero_moser import bench_calogero_moser
from quant_fund.models.kp_hierarchy import bench_kp_hierarchy
from quant_fund.models.nls_soliton import bench_nls_soliton
from quant_fund.models.painleve_eq import bench_painleve_eq
from quant_fund.models.sine_gordon import bench_sine_gordon
from quant_fund.models.toda_lattice import bench_toda_lattice

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


def bench_sine_gordon_family(seed: int = _SEED + 38300) -> dict[str, float]:
    return _finite_blob(bench_sine_gordon(seed))


def bench_nls_soliton_family(seed: int = _SEED + 38301) -> dict[str, float]:
    return _finite_blob(bench_nls_soliton(seed))


def bench_toda_lattice_family(seed: int = _SEED + 38302) -> dict[str, float]:
    return _finite_blob(bench_toda_lattice(seed))


def bench_calogero_moser_family(seed: int = _SEED + 38303) -> dict[str, float]:
    return _finite_blob(bench_calogero_moser(seed))


def bench_kp_hierarchy_family(seed: int = _SEED + 38304) -> dict[str, float]:
    return _finite_blob(bench_kp_hierarchy(seed))


def bench_painleve_eq_family(seed: int = _SEED + 38305) -> dict[str, float]:
    return _finite_blob(bench_painleve_eq(seed))
