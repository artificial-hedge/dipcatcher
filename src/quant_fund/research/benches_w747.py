"""Wave-747 ASEP-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.balazs_seppalainen import (
    bench_balazs_seppalainen,
)
from quant_fund.models.bertini_giacomin import (
    bench_bertini_giacomin,
)
from quant_fund.models.gardina_asym import bench_gardina_asym
from quant_fund.models.quastel_valko import bench_quastel_valko
from quant_fund.models.schutz_tasep import bench_schutz_tasep
from quant_fund.models.timar_tasep import bench_timar_tasep

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_bertini_giacomin_family(
    seed: int = _SEED + 13600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bertini_giacomin",
            bench_bertini_giacomin(seed),
        )
    )


def bench_gardina_asym_family(
    seed: int = _SEED + 13601,
) -> dict[str, float]:
    return _floats(_finite_blob("gardina_asym", bench_gardina_asym(seed)))


def bench_schutz_tasep_family(
    seed: int = _SEED + 13602,
) -> dict[str, float]:
    return _floats(_finite_blob("schutz_tasep", bench_schutz_tasep(seed)))


def bench_balazs_seppalainen_family(
    seed: int = _SEED + 13603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "balazs_seppalainen",
            bench_balazs_seppalainen(seed),
        )
    )


def bench_quastel_valko_family(
    seed: int = _SEED + 13604,
) -> dict[str, float]:
    return _floats(_finite_blob("quastel_valko", bench_quastel_valko(seed)))


def bench_timar_tasep_family(
    seed: int = _SEED + 13605,
) -> dict[str, float]:
    return _floats(_finite_blob("timar_tasep", bench_timar_tasep(seed)))
