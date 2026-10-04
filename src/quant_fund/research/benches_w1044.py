"""Wave-1044 mining-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.blasting_engineering import bench_blasting_engineering
from quant_fund.models.mine_design import bench_mine_design
from quant_fund.models.mine_ventilation import bench_mine_ventilation
from quant_fund.models.mineral_processing import bench_mineral_processing
from quant_fund.models.ore_reserve_estimation import bench_ore_reserve_estimation
from quant_fund.models.rock_mechanics import bench_rock_mechanics

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


def bench_mine_design_family(seed: int = _SEED + 43200) -> dict[str, float]:
    return _finite_blob(bench_mine_design(seed))


def bench_rock_mechanics_family(seed: int = _SEED + 43201) -> dict[str, float]:
    return _finite_blob(bench_rock_mechanics(seed))


def bench_mineral_processing_family(seed: int = _SEED + 43202) -> dict[str, float]:
    return _finite_blob(bench_mineral_processing(seed))


def bench_blasting_engineering_family(seed: int = _SEED + 43203) -> dict[str, float]:
    return _finite_blob(bench_blasting_engineering(seed))


def bench_mine_ventilation_family(seed: int = _SEED + 43204) -> dict[str, float]:
    return _finite_blob(bench_mine_ventilation(seed))


def bench_ore_reserve_estimation_family(seed: int = _SEED + 43205) -> dict[str, float]:
    return _finite_blob(bench_ore_reserve_estimation(seed))
