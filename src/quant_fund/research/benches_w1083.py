"""Wave-1083 humanities-theory canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.hermeneutics import bench_hermeneutics
from quant_fund.models.narratology import bench_narratology
from quant_fund.models.phenomenology import bench_phenomenology
from quant_fund.models.poststructuralism import bench_poststructuralism
from quant_fund.models.semiotics import bench_semiotics
from quant_fund.models.structuralism import bench_structuralism

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


def bench_semiotics_family(seed: int = _SEED + 47100) -> dict[str, float]:
    return _finite_blob(bench_semiotics(seed))


def bench_narratology_family(seed: int = _SEED + 47101) -> dict[str, float]:
    return _finite_blob(bench_narratology(seed))


def bench_hermeneutics_family(seed: int = _SEED + 47102) -> dict[str, float]:
    return _finite_blob(bench_hermeneutics(seed))


def bench_phenomenology_family(seed: int = _SEED + 47103) -> dict[str, float]:
    return _finite_blob(bench_phenomenology(seed))


def bench_structuralism_family(seed: int = _SEED + 47104) -> dict[str, float]:
    return _finite_blob(bench_structuralism(seed))


def bench_poststructuralism_family(seed: int = _SEED + 47105) -> dict[str, float]:
    return _finite_blob(bench_poststructuralism(seed))
