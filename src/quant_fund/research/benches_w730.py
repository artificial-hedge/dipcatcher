"""Wave-730 ramification bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.groth_tame import bench_groth_tame
from quant_fund.models.grothendieck_muw import (
    bench_grothendieck_muw,
)
from quant_fund.models.kato_swan import bench_kato_swan
from quant_fund.models.raynaud_pencil import (
    bench_raynaud_pencil,
)
from quant_fund.models.saito_epsilon import (
    bench_saito_epsilon,
)
from quant_fund.models.swan_conductor import (
    bench_swan_conductor,
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


def bench_grothendieck_muw_family(
    seed: int = _SEED + 11900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "grothendieck_muw",
            bench_grothendieck_muw(seed),
        )
    )


def bench_raynaud_pencil_family(
    seed: int = _SEED + 11901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "raynaud_pencil",
            bench_raynaud_pencil(seed),
        )
    )


def bench_saito_epsilon_family(
    seed: int = _SEED + 11902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "saito_epsilon",
            bench_saito_epsilon(seed),
        )
    )


def bench_swan_conductor_family(
    seed: int = _SEED + 11903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "swan_conductor",
            bench_swan_conductor(seed),
        )
    )


def bench_groth_tame_family(
    seed: int = _SEED + 11904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "groth_tame",
            bench_groth_tame(seed),
        )
    )


def bench_kato_swan_family(
    seed: int = _SEED + 11905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kato_swan",
            bench_kato_swan(seed),
        )
    )
