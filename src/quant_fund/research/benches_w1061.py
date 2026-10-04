"""Wave-1061 law canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.administrative_law import bench_administrative_law
from quant_fund.models.constitutional_law import bench_constitutional_law
from quant_fund.models.contract_law import bench_contract_law
from quant_fund.models.criminal_law import bench_criminal_law
from quant_fund.models.international_law import bench_international_law
from quant_fund.models.tort_law import bench_tort_law

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


def bench_constitutional_law_family(seed: int = _SEED + 44900) -> dict[str, float]:
    return _finite_blob(bench_constitutional_law(seed))


def bench_criminal_law_family(seed: int = _SEED + 44901) -> dict[str, float]:
    return _finite_blob(bench_criminal_law(seed))


def bench_contract_law_family(seed: int = _SEED + 44902) -> dict[str, float]:
    return _finite_blob(bench_contract_law(seed))


def bench_tort_law_family(seed: int = _SEED + 44903) -> dict[str, float]:
    return _finite_blob(bench_tort_law(seed))


def bench_administrative_law_family(seed: int = _SEED + 44904) -> dict[str, float]:
    return _finite_blob(bench_administrative_law(seed))


def bench_international_law_family(seed: int = _SEED + 44905) -> dict[str, float]:
    return _finite_blob(bench_international_law(seed))
