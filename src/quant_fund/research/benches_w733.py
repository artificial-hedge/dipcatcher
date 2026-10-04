"""Wave-733 Hall-algebra-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bridgeland_hall import (
    bench_bridgeland_hall,
)
from quant_fund.models.calaque_hall import bench_calaque_hall
from quant_fund.models.green_hall import bench_green_hall
from quant_fund.models.kontsevich_soibelman import (
    bench_kontsevich_soibelman,
)
from quant_fund.models.morita_hall import bench_morita_hall
from quant_fund.models.mozgovoy_hall import (
    bench_mozgovoy_hall,
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


def bench_green_hall_family(
    seed: int = _SEED + 12200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "green_hall",
            bench_green_hall(seed),
        )
    )


def bench_bridgeland_hall_family(
    seed: int = _SEED + 12201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bridgeland_hall",
            bench_bridgeland_hall(seed),
        )
    )


def bench_kontsevich_soibelman_family(
    seed: int = _SEED + 12202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kontsevich_soibelman",
            bench_kontsevich_soibelman(seed),
        )
    )


def bench_mozgovoy_hall_family(
    seed: int = _SEED + 12203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mozgovoy_hall",
            bench_mozgovoy_hall(seed),
        )
    )


def bench_morita_hall_family(
    seed: int = _SEED + 12204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "morita_hall",
            bench_morita_hall(seed),
        )
    )


def bench_calaque_hall_family(
    seed: int = _SEED + 12205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "calaque_hall",
            bench_calaque_hall(seed),
        )
    )
