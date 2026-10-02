"""Wave-126 adapters: exec-summary deep-learning SOTA — tft_forecaster,
patchtst, lob_transformer, set_transformer, neural_ode, world_model —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.lob_transformer import bench_lob_transformer
from quant_fund.models.neural_ode import bench_neural_ode
from quant_fund.models.patchtst import bench_patchtst
from quant_fund.models.set_transformer import bench_set_transformer
from quant_fund.models.tft_forecaster import bench_tft_forecaster
from quant_fund.models.world_model import bench_world_model

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


def bench_tft_forecaster_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tft_forecaster", bench_tft_forecaster(seed=_SEED + 744)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tft_forecaster bench failed: {exc}") from exc


def bench_patchtst_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("patchtst", bench_patchtst(seed=_SEED + 745)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"patchtst bench failed: {exc}") from exc


def bench_lob_transformer_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lob_transformer", bench_lob_transformer(seed=_SEED + 746)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lob_transformer bench failed: {exc}") from exc


def bench_set_transformer_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("set_transformer", bench_set_transformer(seed=_SEED + 747)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"set_transformer bench failed: {exc}") from exc


def bench_neural_ode_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("neural_ode", bench_neural_ode(seed=_SEED + 748)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neural_ode bench failed: {exc}") from exc


def bench_world_model_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("world_model", bench_world_model(seed=_SEED + 749)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"world_model bench failed: {exc}") from exc
