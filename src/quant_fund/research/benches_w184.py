"""Wave-126 adapters: exec-summary training-dynamics canon — catapult_phase,
hessian_eig, mode_connectivity, ntk_kernel, edge_stability, neural_grok —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.catapult_phase import bench_catapult_phase
from quant_fund.models.edge_stability import bench_edge_stability
from quant_fund.models.hessian_eig import bench_hessian_eig
from quant_fund.models.mode_connectivity import bench_mode_connectivity
from quant_fund.models.neural_grok import bench_neural_grok
from quant_fund.models.ntk_kernel import bench_ntk_kernel

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


def bench_catapult_phase_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("catapult_phase", bench_catapult_phase(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"catapult_phase bench failed: {exc}") from exc


def bench_hessian_eig_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hessian_eig", bench_hessian_eig(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hessian_eig bench failed: {exc}") from exc


def bench_mode_connectivity_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mode_connectivity", bench_mode_connectivity(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mode_connectivity bench failed: {exc}") from exc


def bench_ntk_kernel_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ntk_kernel", bench_ntk_kernel(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ntk_kernel bench failed: {exc}") from exc


def bench_edge_stability_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("edge_stability", bench_edge_stability(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"edge_stability bench failed: {exc}") from exc


def bench_neural_grok_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("neural_grok", bench_neural_grok(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neural_grok bench failed: {exc}") from exc
