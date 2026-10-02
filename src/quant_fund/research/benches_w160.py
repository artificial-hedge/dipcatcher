"""Wave-126 adapters: exec-summary federated-optimization canon — scaffold_fl,
fednova_fl, ditto_fl, moon_fl, fedopt_adam, mime_lite —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ditto_fl import bench_ditto_fl
from quant_fund.models.fednova_fl import bench_fednova_fl
from quant_fund.models.fedopt_adam import bench_fedopt_adam
from quant_fund.models.mime_lite import bench_mime_lite
from quant_fund.models.moon_fl import bench_moon_fl
from quant_fund.models.scaffold_fl import bench_scaffold_fl

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


def bench_scaffold_fl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("scaffold_fl", bench_scaffold_fl(seed=_SEED + 948)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"scaffold_fl bench failed: {exc}") from exc


def bench_fednova_fl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fednova_fl", bench_fednova_fl(seed=_SEED + 949)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fednova_fl bench failed: {exc}") from exc


def bench_ditto_fl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ditto_fl", bench_ditto_fl(seed=_SEED + 950)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ditto_fl bench failed: {exc}") from exc


def bench_moon_fl_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("moon_fl", bench_moon_fl(seed=_SEED + 951)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"moon_fl bench failed: {exc}") from exc


def bench_fedopt_adam_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fedopt_adam", bench_fedopt_adam(seed=_SEED + 952)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fedopt_adam bench failed: {exc}") from exc


def bench_mime_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mime_lite", bench_mime_lite(seed=_SEED + 953)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mime_lite bench failed: {exc}") from exc
