"""Wave-887 solver/transport bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adjoint_sparse import (
    bench_adjoint_sparse,
)
from quant_fund.models.diffusion_approx_sp import (
    bench_diffusion_approx_sp,
)
from quant_fund.models.element_free import (
    bench_element_free,
)
from quant_fund.models.epi_rk import (
    bench_epi_rk,
)
from quant_fund.models.gauss_rk import (
    bench_gauss_rk,
)
from quant_fund.models.importance_rel import (
    bench_importance_rel,
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


def bench_epi_rk_family(
    seed: int = _SEED + 27500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "epi_rk",
            bench_epi_rk(seed),
        )
    )


def bench_gauss_rk_family(
    seed: int = _SEED + 27501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gauss_rk",
            bench_gauss_rk(seed),
        )
    )


def bench_adjoint_sparse_family(
    seed: int = _SEED + 27502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "adjoint_sparse",
            bench_adjoint_sparse(seed),
        )
    )


def bench_element_free_family(
    seed: int = _SEED + 27503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "element_free",
            bench_element_free(seed),
        )
    )


def bench_diffusion_approx_sp_family(
    seed: int = _SEED + 27504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "diffusion_approx_sp",
            bench_diffusion_approx_sp(seed),
        )
    )


def bench_importance_rel_family(
    seed: int = _SEED + 27505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "importance_rel",
            bench_importance_rel(seed),
        )
    )
