"""Wave-976 nuclear-spaces canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.diam_dim import bench_diam_dim
from quant_fund.models.frechet_nuclear import bench_frechet_nuclear
from quant_fund.models.gelfand_triple import bench_gelfand_triple
from quant_fund.models.hilbert_schmidt_emb import bench_hilbert_schmidt_emb
from quant_fund.models.nuclear_map import bench_nuclear_map
from quant_fund.models.trace_duality import bench_trace_duality

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


def bench_nuclear_map_family(seed: int = _SEED + 36400) -> dict[str, float]:
    return _finite_blob(bench_nuclear_map(seed))


def bench_frechet_nuclear_family(seed: int = _SEED + 36401) -> dict[str, float]:
    return _finite_blob(bench_frechet_nuclear(seed))


def bench_gelfand_triple_family(seed: int = _SEED + 36402) -> dict[str, float]:
    return _finite_blob(bench_gelfand_triple(seed))


def bench_hilbert_schmidt_emb_family(seed: int = _SEED + 36403) -> dict[str, float]:
    return _finite_blob(bench_hilbert_schmidt_emb(seed))


def bench_trace_duality_family(seed: int = _SEED + 36404) -> dict[str, float]:
    return _finite_blob(bench_trace_duality(seed))


def bench_diam_dim_family(seed: int = _SEED + 36405) -> dict[str, float]:
    return _finite_blob(bench_diam_dim(seed))
