"""Wave-1131 sociology-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.digital_sociology import bench_digital_sociology
from quant_fund.models.sociology_of_disaster import bench_sociology_of_disaster
from quant_fund.models.sociology_of_housing import bench_sociology_of_housing
from quant_fund.models.sociology_of_migration import bench_sociology_of_migration
from quant_fund.models.sociology_of_risk import bench_sociology_of_risk
from quant_fund.models.sociology_of_the_body import bench_sociology_of_the_body

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


def bench_sociology_of_migration_family(seed: int = _SEED + 51900) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_migration(seed))


def bench_sociology_of_housing_family(seed: int = _SEED + 51901) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_housing(seed))


def bench_sociology_of_disaster_family(seed: int = _SEED + 51902) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_disaster(seed))


def bench_sociology_of_the_body_family(seed: int = _SEED + 51903) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_the_body(seed))


def bench_sociology_of_risk_family(seed: int = _SEED + 51904) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_risk(seed))


def bench_digital_sociology_family(seed: int = _SEED + 51905) -> dict[str, float]:
    return _finite_blob(bench_digital_sociology(seed))
