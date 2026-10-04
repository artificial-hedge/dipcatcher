"""Wave-1063 communications/media canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.communication_theory import bench_communication_theory
from quant_fund.models.digital_media import bench_digital_media
from quant_fund.models.journalism import bench_journalism
from quant_fund.models.media_studies import bench_media_studies
from quant_fund.models.public_relations import bench_public_relations
from quant_fund.models.rhetoric import bench_rhetoric

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


def bench_media_studies_family(seed: int = _SEED + 45100) -> dict[str, float]:
    return _finite_blob(bench_media_studies(seed))


def bench_journalism_family(seed: int = _SEED + 45101) -> dict[str, float]:
    return _finite_blob(bench_journalism(seed))


def bench_public_relations_family(seed: int = _SEED + 45102) -> dict[str, float]:
    return _finite_blob(bench_public_relations(seed))


def bench_rhetoric_family(seed: int = _SEED + 45103) -> dict[str, float]:
    return _finite_blob(bench_rhetoric(seed))


def bench_communication_theory_family(seed: int = _SEED + 45104) -> dict[str, float]:
    return _finite_blob(bench_communication_theory(seed))


def bench_digital_media_family(seed: int = _SEED + 45105) -> dict[str, float]:
    return _finite_blob(bench_digital_media(seed))
