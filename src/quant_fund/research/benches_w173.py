"""Wave-126 adapters: exec-summary diffusion-exotics canon — edm_karras,
rectified_flow, stoch_interp, ddim_ode, cold_diffusion, diff_distill —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cold_diffusion import bench_cold_diffusion
from quant_fund.models.ddim_ode import bench_ddim_ode
from quant_fund.models.diff_distill import bench_diff_distill
from quant_fund.models.edm_karras import bench_edm_karras
from quant_fund.models.rectified_flow import bench_rectified_flow
from quant_fund.models.stoch_interp import bench_stoch_interp

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


def bench_edm_karras_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("edm_karras", bench_edm_karras(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"edm_karras bench failed: {exc}") from exc


def bench_rectified_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rectified_flow", bench_rectified_flow(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rectified_flow bench failed: {exc}") from exc


def bench_stoch_interp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("stoch_interp", bench_stoch_interp(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"stoch_interp bench failed: {exc}") from exc


def bench_ddim_ode_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ddim_ode", bench_ddim_ode(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ddim_ode bench failed: {exc}") from exc


def bench_cold_diffusion_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cold_diffusion", bench_cold_diffusion(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cold_diffusion bench failed: {exc}") from exc


def bench_diff_distill_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("diff_distill", bench_diff_distill(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"diff_distill bench failed: {exc}") from exc
