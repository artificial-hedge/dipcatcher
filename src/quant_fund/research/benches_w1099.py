"""Wave-1099 psychology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.abnormal_psychology import bench_abnormal_psychology
from quant_fund.models.forensic_psychology import bench_forensic_psychology
from quant_fund.models.health_psychology import bench_health_psychology
from quant_fund.models.neuropsychology import bench_neuropsychology
from quant_fund.models.organizational_psychology import bench_organizational_psychology
from quant_fund.models.personality_psychology import bench_personality_psychology

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


def bench_personality_psychology_family(seed: int = _SEED + 48700) -> dict[str, float]:
    return _finite_blob(bench_personality_psychology(seed))


def bench_abnormal_psychology_family(seed: int = _SEED + 48701) -> dict[str, float]:
    return _finite_blob(bench_abnormal_psychology(seed))


def bench_health_psychology_family(seed: int = _SEED + 48702) -> dict[str, float]:
    return _finite_blob(bench_health_psychology(seed))


def bench_neuropsychology_family(seed: int = _SEED + 48703) -> dict[str, float]:
    return _finite_blob(bench_neuropsychology(seed))


def bench_forensic_psychology_family(seed: int = _SEED + 48704) -> dict[str, float]:
    return _finite_blob(bench_forensic_psychology(seed))


def bench_organizational_psychology_family(seed: int = _SEED + 48705) -> dict[str, float]:
    return _finite_blob(bench_organizational_psychology(seed))
