"""Wave-815 excursion-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.excursion_proc import (
    bench_excursion_proc,
)
from quant_fund.models.inverse_local import (
    bench_inverse_local,
)
from quant_fund.models.knight_theorem import (
    bench_knight_theorem,
)
from quant_fund.models.mazza_yor import (
    bench_mazza_yor,
)
from quant_fund.models.pitman_thm import (
    bench_pitman_thm,
)
from quant_fund.models.ray_knight import (
    bench_ray_knight,
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


def bench_excursion_proc_family(
    seed: int = _SEED + 20300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "excursion_proc",
            bench_excursion_proc(seed),
        )
    )


def bench_inverse_local_family(
    seed: int = _SEED + 20301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "inverse_local",
            bench_inverse_local(seed),
        )
    )


def bench_ray_knight_family(
    seed: int = _SEED + 20302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ray_knight",
            bench_ray_knight(seed),
        )
    )


def bench_knight_theorem_family(
    seed: int = _SEED + 20303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "knight_theorem",
            bench_knight_theorem(seed),
        )
    )


def bench_mazza_yor_family(
    seed: int = _SEED + 20304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mazza_yor",
            bench_mazza_yor(seed),
        )
    )


def bench_pitman_thm_family(
    seed: int = _SEED + 20305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pitman_thm",
            bench_pitman_thm(seed),
        )
    )
