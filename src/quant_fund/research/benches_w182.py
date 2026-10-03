"""Wave-126 adapters: exec-summary normalizing-flow canon — real_nvp,
glow_flow, neural_spline_flow, maf_flow, planar_flow, iaf_flow —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.glow_flow import bench_glow_flow
from quant_fund.models.iaf_flow import bench_iaf_flow
from quant_fund.models.maf_flow import bench_maf_flow
from quant_fund.models.neural_spline_flow import bench_neural_spline_flow
from quant_fund.models.planar_flow import bench_planar_flow
from quant_fund.models.real_nvp import bench_real_nvp

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


def bench_real_nvp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("real_nvp", bench_real_nvp(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"real_nvp bench failed: {exc}") from exc


def bench_glow_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("glow_flow", bench_glow_flow(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"glow_flow bench failed: {exc}") from exc


def bench_neural_spline_flow_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("neural_spline_flow", bench_neural_spline_flow(seed=_SEED + 962))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neural_spline_flow bench failed: {exc}") from exc


def bench_maf_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("maf_flow", bench_maf_flow(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"maf_flow bench failed: {exc}") from exc


def bench_planar_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("planar_flow", bench_planar_flow(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"planar_flow bench failed: {exc}") from exc


def bench_iaf_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("iaf_flow", bench_iaf_flow(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"iaf_flow bench failed: {exc}") from exc
