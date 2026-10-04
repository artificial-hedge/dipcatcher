"""Wave-1147 biomedical-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.anatomy import bench_anatomy
from quant_fund.models.cardiology_2 import bench_cardiology_2
from quant_fund.models.endocrinology_2 import bench_endocrinology_2
from quant_fund.models.immunology_2 import bench_immunology_2
from quant_fund.models.neuroscience_2 import bench_neuroscience_2
from quant_fund.models.physiology_2 import bench_physiology_2

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


def bench_anatomy_family(seed: int = _SEED + 53500) -> dict[str, float]:
    return _finite_blob(bench_anatomy(seed))


def bench_physiology_2_family(seed: int = _SEED + 53501) -> dict[str, float]:
    return _finite_blob(bench_physiology_2(seed))


def bench_endocrinology_2_family(seed: int = _SEED + 53502) -> dict[str, float]:
    return _finite_blob(bench_endocrinology_2(seed))


def bench_neuroscience_2_family(seed: int = _SEED + 53503) -> dict[str, float]:
    return _finite_blob(bench_neuroscience_2(seed))


def bench_cardiology_2_family(seed: int = _SEED + 53504) -> dict[str, float]:
    return _finite_blob(bench_cardiology_2(seed))


def bench_immunology_2_family(seed: int = _SEED + 53505) -> dict[str, float]:
    return _finite_blob(bench_immunology_2(seed))
