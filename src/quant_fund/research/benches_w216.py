"""Wave-216 adapters: advanced-MC-sampling canon — parallel_tempering,
wang_landau, umbrella_sampling, metadynamics, wham, thermo_integration —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.metadynamics import bench_metadynamics
from quant_fund.models.parallel_tempering import bench_parallel_tempering
from quant_fund.models.thermo_integration import bench_thermo_integration
from quant_fund.models.umbrella_sampling import bench_umbrella_sampling
from quant_fund.models.wang_landau import bench_wang_landau
from quant_fund.models.wham import bench_wham

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


def bench_wham_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("wham", bench_wham(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"wham bench failed: {exc}") from exc


def bench_parallel_tempering_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("parallel_tempering", bench_parallel_tempering(seed=_SEED + 961))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"parallel_tempering bench failed: {exc}") from exc


def bench_metadynamics_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("metadynamics", bench_metadynamics(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"metadynamics bench failed: {exc}") from exc


def bench_wang_landau_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("wang_landau", bench_wang_landau(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"wang_landau bench failed: {exc}") from exc


def bench_umbrella_sampling_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("umbrella_sampling", bench_umbrella_sampling(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"umbrella_sampling bench failed: {exc}") from exc


def bench_thermo_integration_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("thermo_integration", bench_thermo_integration(seed=_SEED + 965))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"thermo_integration bench failed: {exc}") from exc
