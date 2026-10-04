"""Wave-861 time-marching/ODE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ars_imex import (
    bench_ars_imex,
)
from quant_fund.models.dirk_scheme import (
    bench_dirk_scheme,
)
from quant_fund.models.exponential_euler import (
    bench_exponential_euler,
)
from quant_fund.models.imex_rk import (
    bench_imex_rk,
)
from quant_fund.models.rosenbrock_w import (
    bench_rosenbrock_w,
)
from quant_fund.models.ssp_rk import (
    bench_ssp_rk,
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


def bench_imex_rk_family(
    seed: int = _SEED + 24900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "imex_rk",
            bench_imex_rk(seed),
        )
    )


def bench_ssp_rk_family(
    seed: int = _SEED + 24901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ssp_rk",
            bench_ssp_rk(seed),
        )
    )


def bench_exponential_euler_family(
    seed: int = _SEED + 24902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "exponential_euler",
            bench_exponential_euler(seed),
        )
    )


def bench_rosenbrock_w_family(
    seed: int = _SEED + 24903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rosenbrock_w",
            bench_rosenbrock_w(seed),
        )
    )


def bench_ars_imex_family(
    seed: int = _SEED + 24904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ars_imex",
            bench_ars_imex(seed),
        )
    )


def bench_dirk_scheme_family(
    seed: int = _SEED + 24905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dirk_scheme",
            bench_dirk_scheme(seed),
        )
    )
