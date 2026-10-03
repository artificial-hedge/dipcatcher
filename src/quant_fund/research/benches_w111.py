"""Wave-111 adapters: computational-geometry canon — Kabsch
rigid alignment, point-to-point ICP, discrete Fréchet, Hausdorff
(directed/symmetric/robust), Graham-scan convex hull, and
Bowyer–Watson Delaunay — each benched on SYNTHETIC sets with
closed-form or scipy-verified references. Adapters flatten to
a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.convex_hull import bench_convex_hull
from quant_fund.models.delaunay import bench_delaunay
from quant_fund.models.frechet import bench_frechet
from quant_fund.models.hausdorff import bench_hausdorff
from quant_fund.models.icp import bench_icp
from quant_fund.models.kabsch import bench_kabsch

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
                flat[f"{k}_{i}"] = f
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_kabsch_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("kabsch", bench_kabsch(seed=_SEED + 654)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kabsch bench failed: {exc}") from exc


def bench_icp_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("icp", bench_icp(seed=_SEED + 655)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"icp bench failed: {exc}") from exc


def bench_frechet_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("frechet", bench_frechet(seed=_SEED + 656)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"frechet bench failed: {exc}") from exc


def bench_hausdorff_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("hausdorff", bench_hausdorff(seed=_SEED + 657)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hausdorff bench failed: {exc}") from exc


def bench_convex_hull_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("convex_hull", bench_convex_hull(seed=_SEED + 658)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"convex_hull bench failed: {exc}") from exc


def bench_delaunay_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("delaunay", bench_delaunay(seed=_SEED + 659)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"delaunay bench failed: {exc}") from exc
