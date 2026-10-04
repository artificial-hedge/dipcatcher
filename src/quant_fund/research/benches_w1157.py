"""Wave-1157 computing-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.artificial_intelligence import bench_artificial_intelligence
from quant_fund.models.computer_science_2 import bench_computer_science_2
from quant_fund.models.data_engineering import bench_data_engineering
from quant_fund.models.information_theory_2 import bench_information_theory_2
from quant_fund.models.machine_learning_2 import bench_machine_learning_2
from quant_fund.models.software_engineering import bench_software_engineering

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


def bench_computer_science_2_family(seed: int = _SEED + 54500) -> dict[str, float]:
    return _finite_blob(bench_computer_science_2(seed))


def bench_software_engineering_family(seed: int = _SEED + 54501) -> dict[str, float]:
    return _finite_blob(bench_software_engineering(seed))


def bench_machine_learning_2_family(seed: int = _SEED + 54502) -> dict[str, float]:
    return _finite_blob(bench_machine_learning_2(seed))


def bench_artificial_intelligence_family(seed: int = _SEED + 54503) -> dict[str, float]:
    return _finite_blob(bench_artificial_intelligence(seed))


def bench_data_engineering_family(seed: int = _SEED + 54504) -> dict[str, float]:
    return _finite_blob(bench_data_engineering(seed))


def bench_information_theory_2_family(seed: int = _SEED + 54505) -> dict[str, float]:
    return _finite_blob(bench_information_theory_2(seed))
