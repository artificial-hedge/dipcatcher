"""Wave-126 adapters: exec-summary NAS canon — random_search_nas,
evolution_nas, darts_nas, enas_controller, one_shot_nas, arch_predictor —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arch_predictor import bench_arch_predictor
from quant_fund.models.darts_nas import bench_darts_nas
from quant_fund.models.enas_controller import bench_enas_controller
from quant_fund.models.evolution_nas import bench_evolution_nas
from quant_fund.models.one_shot_nas import bench_one_shot_nas
from quant_fund.models.random_search_nas import bench_random_search_nas

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


def bench_random_search_nas_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("random_search_nas", bench_random_search_nas(seed=_SEED + 906)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"random_search_nas bench failed: {exc}") from exc


def bench_evolution_nas_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("evolution_nas", bench_evolution_nas(seed=_SEED + 907)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"evolution_nas bench failed: {exc}") from exc


def bench_darts_nas_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("darts_nas", bench_darts_nas(seed=_SEED + 908)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"darts_nas bench failed: {exc}") from exc


def bench_enas_controller_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("enas_controller", bench_enas_controller(seed=_SEED + 909)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"enas_controller bench failed: {exc}") from exc


def bench_one_shot_nas_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("one_shot_nas", bench_one_shot_nas(seed=_SEED + 910)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"one_shot_nas bench failed: {exc}") from exc


def bench_arch_predictor_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("arch_predictor", bench_arch_predictor(seed=_SEED + 911)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"arch_predictor bench failed: {exc}") from exc
