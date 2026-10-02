"""Wave-126 adapters: exec-summary scientific-ML-PDE canon — moc_lines,
deepritz_pinn, spectral_pde, weak_form_pinn, fbsde_solver, feynman_kac_mc —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.deepritz_pinn import bench_deepritz_pinn
from quant_fund.models.fbsde_solver import bench_fbsde_solver
from quant_fund.models.feynman_kac_mc import bench_feynman_kac_mc
from quant_fund.models.moc_lines import bench_moc_lines
from quant_fund.models.spectral_pde import bench_spectral_pde
from quant_fund.models.weak_form_pinn import bench_weak_form_pinn

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


def bench_moc_lines_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("moc_lines", bench_moc_lines(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"moc_lines bench failed: {exc}") from exc


def bench_deepritz_pinn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deepritz_pinn", bench_deepritz_pinn(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deepritz_pinn bench failed: {exc}") from exc


def bench_spectral_pde_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("spectral_pde", bench_spectral_pde(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"spectral_pde bench failed: {exc}") from exc


def bench_weak_form_pinn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("weak_form_pinn", bench_weak_form_pinn(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"weak_form_pinn bench failed: {exc}") from exc


def bench_fbsde_solver_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fbsde_solver", bench_fbsde_solver(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fbsde_solver bench failed: {exc}") from exc


def bench_feynman_kac_mc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("feynman_kac_mc", bench_feynman_kac_mc(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"feynman_kac_mc bench failed: {exc}") from exc
