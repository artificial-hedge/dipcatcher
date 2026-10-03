"""Wave-126 adapters: exec-summary conditional-density canon — mdn_cond,
flow_regression, diffusion_regressor, het_gp, crps_net, kernel_mixture —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.crps_net import bench_crps_net
from quant_fund.models.diffusion_regressor import bench_diffusion_regressor
from quant_fund.models.flow_regression import bench_flow_regression
from quant_fund.models.het_gp import bench_het_gp
from quant_fund.models.kernel_mixture import bench_kernel_mixture
from quant_fund.models.mdn_cond import bench_mdn_cond

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


def bench_mdn_cond_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mdn_cond", bench_mdn_cond(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mdn_cond bench failed: {exc}") from exc


def bench_flow_regression_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("flow_regression", bench_flow_regression(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"flow_regression bench failed: {exc}") from exc


def bench_diffusion_regressor_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("diffusion_regressor", bench_diffusion_regressor(seed=_SEED + 962))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"diffusion_regressor bench failed: {exc}") from exc


def bench_het_gp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("het_gp", bench_het_gp(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"het_gp bench failed: {exc}") from exc


def bench_crps_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("crps_net", bench_crps_net(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crps_net bench failed: {exc}") from exc


def bench_kernel_mixture_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("kernel_mixture", bench_kernel_mixture(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kernel_mixture bench failed: {exc}") from exc
