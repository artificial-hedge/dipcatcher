"""Wave-879 exponential-time-integrator bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.etd_rk4_classic import (
    bench_etd_rk4_classic,
)
from quant_fund.models.expm_int import (
    bench_expm_int,
)
from quant_fund.models.expokit import (
    bench_expokit,
)
from quant_fund.models.krylov_subspace_time import (
    bench_krylov_subspace_time,
)
from quant_fund.models.leja_point import (
    bench_leja_point,
)
from quant_fund.models.phi_function import (
    bench_phi_function,
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


def bench_expm_int_family(
    seed: int = _SEED + 26700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "expm_int",
            bench_expm_int(seed),
        )
    )


def bench_expokit_family(
    seed: int = _SEED + 26701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "expokit",
            bench_expokit(seed),
        )
    )


def bench_krylov_subspace_time_family(
    seed: int = _SEED + 26702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "krylov_subspace_time",
            bench_krylov_subspace_time(seed),
        )
    )


def bench_leja_point_family(
    seed: int = _SEED + 26703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "leja_point",
            bench_leja_point(seed),
        )
    )


def bench_phi_function_family(
    seed: int = _SEED + 26704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "phi_function",
            bench_phi_function(seed),
        )
    )


def bench_etd_rk4_classic_family(
    seed: int = _SEED + 26705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etd_rk4_classic",
            bench_etd_rk4_classic(seed),
        )
    )
