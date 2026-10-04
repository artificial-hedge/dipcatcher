"""Wave-799 path-PDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dupire_functional import (
    bench_dupire_functional,
)
from quant_fund.models.functional_ito import (
    bench_functional_ito,
)
from quant_fund.models.kolmogorov_path import (
    bench_kolmogorov_path,
)
from quant_fund.models.path_dependent_pde import (
    bench_path_dependent_pde,
)
from quant_fund.models.path_sobolev import (
    bench_path_sobolev,
)
from quant_fund.models.viscosity_path import (
    bench_viscosity_path,
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


def bench_path_dependent_pde_family(
    seed: int = _SEED + 18800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "path_dependent_pde",
            bench_path_dependent_pde(seed),
        )
    )


def bench_functional_ito_family(
    seed: int = _SEED + 18801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "functional_ito",
            bench_functional_ito(seed),
        )
    )


def bench_dupire_functional_family(
    seed: int = _SEED + 18802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dupire_functional",
            bench_dupire_functional(seed),
        )
    )


def bench_viscosity_path_family(
    seed: int = _SEED + 18803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "viscosity_path",
            bench_viscosity_path(seed),
        )
    )


def bench_path_sobolev_family(
    seed: int = _SEED + 18804,
) -> dict[str, float]:
    return _floats(_finite_blob("path_sobolev", bench_path_sobolev(seed)))


def bench_kolmogorov_path_family(
    seed: int = _SEED + 18805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kolmogorov_path",
            bench_kolmogorov_path(seed),
        )
    )
