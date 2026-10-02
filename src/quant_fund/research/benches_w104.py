"""Wave-104 adapters: optimal-transport canon II — log-
domain Sinkhorn vs exact transport LP, 1-D/n-D EMD +
Bures-Wasserstein, entropic Gromov-Wasserstein, KL-relaxed
unbalanced OT, fixed-support Wasserstein barycenter, and
fused Gromov-Wasserstein.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.emd_lp import bench_emd_lp
from quant_fund.models.fused_gromov import bench_fused_gromov
from quant_fund.models.gromov_wasserstein import bench_gromov_wasserstein
from quant_fund.models.sinkhorn import bench_sinkhorn
from quant_fund.models.unbalanced_ot import bench_unbalanced_ot
from quant_fund.models.wasserstein_barycenter import bench_wasserstein_barycenter

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


def bench_sinkhorn_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("sinkhorn", bench_sinkhorn(seed=_SEED + 612)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sinkhorn bench failed: {exc}") from exc


def bench_emd_lp_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("emd_lp", bench_emd_lp(seed=_SEED + 613)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"emd_lp bench failed: {exc}") from exc


def bench_gromov_wasserstein_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("gromov_wasserstein", bench_gromov_wasserstein(seed=_SEED + 614))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gromov_wasserstein bench failed: {exc}") from exc


def bench_unbalanced_ot_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("unbalanced_ot", bench_unbalanced_ot(seed=_SEED + 615))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"unbalanced_ot bench failed: {exc}") from exc


def bench_wasserstein_barycenter_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob(
                "wasserstein_barycenter",
                bench_wasserstein_barycenter(seed=_SEED + 616),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"wasserstein_barycenter bench failed: {exc}") from exc


def bench_fused_gromov_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("fused_gromov", bench_fused_gromov(seed=_SEED + 617))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fused_gromov bench failed: {exc}") from exc
