"""Wave-1170 justice canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.criminology_3 import bench_criminology_3
from quant_fund.models.forensic_science_2 import bench_forensic_science_2
from quant_fund.models.intelligence_studies_2 import bench_intelligence_studies_2
from quant_fund.models.penology_2 import bench_penology_2
from quant_fund.models.security_studies_2 import bench_security_studies_2
from quant_fund.models.victimology_2 import bench_victimology_2

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


def bench_criminology_3_family(seed: int = _SEED + 55800) -> dict[str, float]:
    return _finite_blob(bench_criminology_3(seed))


def bench_forensic_science_2_family(seed: int = _SEED + 55801) -> dict[str, float]:
    return _finite_blob(bench_forensic_science_2(seed))


def bench_penology_2_family(seed: int = _SEED + 55802) -> dict[str, float]:
    return _finite_blob(bench_penology_2(seed))


def bench_victimology_2_family(seed: int = _SEED + 55803) -> dict[str, float]:
    return _finite_blob(bench_victimology_2(seed))


def bench_security_studies_2_family(seed: int = _SEED + 55804) -> dict[str, float]:
    return _finite_blob(bench_security_studies_2(seed))


def bench_intelligence_studies_2_family(seed: int = _SEED + 55805) -> dict[str, float]:
    return _finite_blob(bench_intelligence_studies_2(seed))
