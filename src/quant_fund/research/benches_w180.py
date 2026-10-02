"""Wave-126 adapters: exec-summary PDMP exotic-sampling canon — bouncy_particle,
zigzag_sampler, boomerang_sampler, kinetic_langevin, elliptical_slice, riemannian_mala —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boomerang_sampler import bench_boomerang_sampler
from quant_fund.models.bouncy_particle import bench_bouncy_particle
from quant_fund.models.elliptical_slice import bench_elliptical_slice
from quant_fund.models.kinetic_langevin import bench_kinetic_langevin
from quant_fund.models.riemannian_mala import bench_riemannian_mala
from quant_fund.models.zigzag_sampler import bench_zigzag_sampler

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


def bench_bouncy_particle_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bouncy_particle", bench_bouncy_particle(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bouncy_particle bench failed: {exc}") from exc


def bench_zigzag_sampler_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("zigzag_sampler", bench_zigzag_sampler(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"zigzag_sampler bench failed: {exc}") from exc


def bench_boomerang_sampler_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("boomerang_sampler", bench_boomerang_sampler(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"boomerang_sampler bench failed: {exc}") from exc


def bench_kinetic_langevin_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("kinetic_langevin", bench_kinetic_langevin(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kinetic_langevin bench failed: {exc}") from exc


def bench_elliptical_slice_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("elliptical_slice", bench_elliptical_slice(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"elliptical_slice bench failed: {exc}") from exc


def bench_riemannian_mala_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("riemannian_mala", bench_riemannian_mala(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"riemannian_mala bench failed: {exc}") from exc
