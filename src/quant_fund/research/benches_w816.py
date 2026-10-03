"""Wave-816 Markov-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cadlag_markov import (
    bench_cadlag_markov,
)
from quant_fund.models.characteristic_markov import (
    bench_characteristic_markov,
)
from quant_fund.models.generator_markov import (
    bench_generator_markov,
)
from quant_fund.models.hunt_process import (
    bench_hunt_process,
)
from quant_fund.models.resolvent_markov import (
    bench_resolvent_markov,
)
from quant_fund.models.transition_semigroup import (
    bench_transition_semigroup,
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


def bench_hunt_process_family(
    seed: int = _SEED + 20400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hunt_process",
            bench_hunt_process(seed),
        )
    )


def bench_cadlag_markov_family(
    seed: int = _SEED + 20401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cadlag_markov",
            bench_cadlag_markov(seed),
        )
    )


def bench_transition_semigroup_family(
    seed: int = _SEED + 20402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "transition_semigroup",
            bench_transition_semigroup(seed),
        )
    )


def bench_resolvent_markov_family(
    seed: int = _SEED + 20403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "resolvent_markov",
            bench_resolvent_markov(seed),
        )
    )


def bench_generator_markov_family(
    seed: int = _SEED + 20404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "generator_markov",
            bench_generator_markov(seed),
        )
    )


def bench_characteristic_markov_family(
    seed: int = _SEED + 20405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "characteristic_markov",
            bench_characteristic_markov(seed),
        )
    )
