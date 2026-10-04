"""Wave-521 analytic-NT bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chebyshev_bias import bench_chebyshev_bias
from quant_fund.models.dirichlet_l import bench_dirichlet_l
from quant_fund.models.explicit_formula import bench_explicit_formula
from quant_fund.models.linnik_thm import bench_linnik_thm
from quant_fund.models.riemann_zeta import bench_riemann_zeta
from quant_fund.models.zero_density import bench_zero_density

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


def bench_explicit_formula_family(seed: int = _SEED + 3032) -> dict[str, float]:
    return _floats(_finite_blob("explicit_formula", bench_explicit_formula(seed)))


def bench_zero_density_family(seed: int = _SEED + 3033) -> dict[str, float]:
    return _floats(_finite_blob("zero_density", bench_zero_density(seed)))


def bench_riemann_zeta_family(seed: int = _SEED + 3034) -> dict[str, float]:
    return _floats(_finite_blob("riemann_zeta", bench_riemann_zeta(seed)))


def bench_dirichlet_l_family(seed: int = _SEED + 3035) -> dict[str, float]:
    return _floats(_finite_blob("dirichlet_l", bench_dirichlet_l(seed)))


def bench_linnik_thm_family(seed: int = _SEED + 3036) -> dict[str, float]:
    return _floats(_finite_blob("linnik_thm", bench_linnik_thm(seed)))


def bench_chebyshev_bias_family(seed: int = _SEED + 3037) -> dict[str, float]:
    return _floats(_finite_blob("chebyshev_bias", bench_chebyshev_bias(seed)))
