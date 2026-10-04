"""Wave-808 filtering bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bene_filter import (
    bench_bene_filter,
)
from quant_fund.models.hidden_markov_filter import (
    bench_hidden_markov_filter,
)
from quant_fund.models.kalman_bucy import (
    bench_kalman_bucy,
)
from quant_fund.models.kushner_strat import (
    bench_kushner_strat,
)
from quant_fund.models.particle_filter2 import (
    bench_particle_filter2,
)
from quant_fund.models.zakai_eq import (
    bench_zakai_eq,
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


def bench_zakai_eq_family(
    seed: int = _SEED + 19700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zakai_eq",
            bench_zakai_eq(seed),
        )
    )


def bench_kushner_strat_family(
    seed: int = _SEED + 19701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kushner_strat",
            bench_kushner_strat(seed),
        )
    )


def bench_kalman_bucy_family(
    seed: int = _SEED + 19702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kalman_bucy",
            bench_kalman_bucy(seed),
        )
    )


def bench_bene_filter_family(
    seed: int = _SEED + 19703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bene_filter",
            bench_bene_filter(seed),
        )
    )


def bench_hidden_markov_filter_family(
    seed: int = _SEED + 19704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hidden_markov_filter",
            bench_hidden_markov_filter(seed),
        )
    )


def bench_particle_filter2_family(
    seed: int = _SEED + 19705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "particle_filter2",
            bench_particle_filter2(seed),
        )
    )
