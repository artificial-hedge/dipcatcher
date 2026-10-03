"""Wave-821 random-measure bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.compensator_rm import (
    bench_compensator_rm,
)
from quant_fund.models.integer_measure import (
    bench_integer_measure,
)
from quant_fund.models.jump_measure import (
    bench_jump_measure,
)
from quant_fund.models.poisson_rm import (
    bench_poisson_rm,
)
from quant_fund.models.random_measure import (
    bench_random_measure,
)
from quant_fund.models.sato_measure import (
    bench_sato_measure,
)

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


def bench_random_measure_family(
    seed: int = _SEED + 20900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "random_measure",
            bench_random_measure(seed),
        )
    )


def bench_integer_measure_family(
    seed: int = _SEED + 20901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "integer_measure",
            bench_integer_measure(seed),
        )
    )


def bench_poisson_rm_family(
    seed: int = _SEED + 20902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "poisson_rm",
            bench_poisson_rm(seed),
        )
    )


def bench_compensator_rm_family(
    seed: int = _SEED + 20903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "compensator_rm",
            bench_compensator_rm(seed),
        )
    )


def bench_jump_measure_family(
    seed: int = _SEED + 20904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jump_measure",
            bench_jump_measure(seed),
        )
    )


def bench_sato_measure_family(
    seed: int = _SEED + 20905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sato_measure",
            bench_sato_measure(seed),
        )
    )
