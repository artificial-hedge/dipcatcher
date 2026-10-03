"""Wave-221 adapters: computational-algebra canon — buchberger, resultant,
poly_gcd, gf2_factor, lll_reduce, newton_interp —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.buchberger import bench_buchberger
from quant_fund.models.gf2_factor import bench_gf2_factor
from quant_fund.models.lll_reduce import bench_lll_reduce
from quant_fund.models.newton_interp import bench_newton_interp
from quant_fund.models.poly_gcd import bench_poly_gcd
from quant_fund.models.resultant import bench_resultant

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_buchberger_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("buchberger", bench_buchberger(seed=_SEED + 980)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"buchberger bench failed: {exc}") from exc


def bench_gf2_factor_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gf2_factor", bench_gf2_factor(seed=_SEED + 981)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gf2_factor bench failed: {exc}") from exc


def bench_lll_reduce_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lll_reduce", bench_lll_reduce(seed=_SEED + 982)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lll_reduce bench failed: {exc}") from exc


def bench_newton_interp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("newton_interp", bench_newton_interp(seed=_SEED + 983)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"newton_interp bench failed: {exc}") from exc


def bench_poly_gcd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("poly_gcd", bench_poly_gcd(seed=_SEED + 984)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"poly_gcd bench failed: {exc}") from exc


def bench_resultant_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("resultant", bench_resultant(seed=_SEED + 985)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"resultant bench failed: {exc}") from exc
