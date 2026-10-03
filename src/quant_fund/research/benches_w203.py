"""Wave-203 adapters: information-geometry canon — fisher_rao,
natural_gradient, mirror_descent, bregman_nmf, alpha_geodesic, jko_scheme —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alpha_geodesic import bench_alpha_geodesic
from quant_fund.models.bregman_nmf import bench_bregman_nmf
from quant_fund.models.fisher_rao import bench_fisher_rao
from quant_fund.models.jko_scheme import bench_jko_scheme
from quant_fund.models.mirror_descent import bench_mirror_descent
from quant_fund.models.natural_gradient import bench_natural_gradient

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


def bench_alpha_geodesic_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("alpha_geodesic", bench_alpha_geodesic(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"alpha_geodesic bench failed: {exc}") from exc


def bench_fisher_rao_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fisher_rao", bench_fisher_rao(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fisher_rao bench failed: {exc}") from exc


def bench_bregman_nmf_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bregman_nmf", bench_bregman_nmf(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bregman_nmf bench failed: {exc}") from exc


def bench_natural_gradient_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("natural_gradient", bench_natural_gradient(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"natural_gradient bench failed: {exc}") from exc


def bench_mirror_descent_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mirror_descent", bench_mirror_descent(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mirror_descent bench failed: {exc}") from exc


def bench_jko_scheme_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("jko_scheme", bench_jko_scheme(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"jko_scheme bench failed: {exc}") from exc
