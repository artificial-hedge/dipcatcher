"""Wave-1206 transplantation canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.immunosuppression import bench_immunosuppression
from quant_fund.models.organ_donation import bench_organ_donation
from quant_fund.models.regenerative_medicine import bench_regenerative_medicine
from quant_fund.models.stem_cell_therapy import bench_stem_cell_therapy
from quant_fund.models.transplantation_medicine import bench_transplantation_medicine
from quant_fund.models.xenotransplantation import bench_xenotransplantation

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


def bench_transplantation_medicine_family(seed: int = _SEED + 59400) -> dict[str, float]:
    return _finite_blob(bench_transplantation_medicine(seed))


def bench_organ_donation_family(seed: int = _SEED + 59401) -> dict[str, float]:
    return _finite_blob(bench_organ_donation(seed))


def bench_immunosuppression_family(seed: int = _SEED + 59402) -> dict[str, float]:
    return _finite_blob(bench_immunosuppression(seed))


def bench_xenotransplantation_family(seed: int = _SEED + 59403) -> dict[str, float]:
    return _finite_blob(bench_xenotransplantation(seed))


def bench_stem_cell_therapy_family(seed: int = _SEED + 59404) -> dict[str, float]:
    return _finite_blob(bench_stem_cell_therapy(seed))


def bench_regenerative_medicine_family(seed: int = _SEED + 59405) -> dict[str, float]:
    return _finite_blob(bench_regenerative_medicine(seed))
