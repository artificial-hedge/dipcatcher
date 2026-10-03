"""Wave-655 homotopy-21 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bo_htpy import bench_bo_htpy
from quant_fund.models.chromatic_square import bench_chromatic_square
from quant_fund.models.devinatz_htpy import bench_devinatz_htpy
from quant_fund.models.hopkins_smith import bench_hopkins_smith
from quant_fund.models.morava_stab import bench_morava_stab
from quant_fund.models.telescope_tower import bench_telescope_tower

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


def bench_devinatz_htpy_family(
    seed: int = _SEED + 4400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "devinatz_htpy",
            bench_devinatz_htpy(seed),
        )
    )


def bench_hopkins_smith_family(
    seed: int = _SEED + 4401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hopkins_smith",
            bench_hopkins_smith(seed),
        )
    )


def bench_morava_stab_family(
    seed: int = _SEED + 4402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "morava_stab",
            bench_morava_stab(seed),
        )
    )


def bench_chromatic_square_family(
    seed: int = _SEED + 4403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chromatic_square",
            bench_chromatic_square(seed),
        )
    )


def bench_telescope_tower_family(
    seed: int = _SEED + 4404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "telescope_tower",
            bench_telescope_tower(seed),
        )
    )


def bench_bo_htpy_family(
    seed: int = _SEED + 4405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bo_htpy",
            bench_bo_htpy(seed),
        )
    )
