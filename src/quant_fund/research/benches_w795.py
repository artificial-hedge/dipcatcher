"""Wave-795 stochastic-games bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.differential_game import (
    bench_differential_game,
)
from quant_fund.models.dynkin_game import (
    bench_dynkin_game,
)
from quant_fund.models.isaacs_equation import (
    bench_isaacs_equation,
)
from quant_fund.models.nonzero_sum_game import (
    bench_nonzero_sum_game,
)
from quant_fund.models.stochastic_game2 import (
    bench_stochastic_game2,
)
from quant_fund.models.zero_sum_game import (
    bench_zero_sum_game,
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


def bench_dynkin_game_family(
    seed: int = _SEED + 18400,
) -> dict[str, float]:
    return _floats(_finite_blob("dynkin_game", bench_dynkin_game(seed)))


def bench_stochastic_game2_family(
    seed: int = _SEED + 18401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_game2",
            bench_stochastic_game2(seed),
        )
    )


def bench_differential_game_family(
    seed: int = _SEED + 18402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "differential_game",
            bench_differential_game(seed),
        )
    )


def bench_zero_sum_game_family(
    seed: int = _SEED + 18403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zero_sum_game",
            bench_zero_sum_game(seed),
        )
    )


def bench_nonzero_sum_game_family(
    seed: int = _SEED + 18404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nonzero_sum_game",
            bench_nonzero_sum_game(seed),
        )
    )


def bench_isaacs_equation_family(
    seed: int = _SEED + 18405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "isaacs_equation",
            bench_isaacs_equation(seed),
        )
    )
