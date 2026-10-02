"""Wave-126 adapters: exec-summary OOD-detection canon — mahalanobis_ood,
max_softmax_ood, gradient_norm_ood, energy_ood, knn_ood, vim_ood —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.energy_ood import bench_energy_ood
from quant_fund.models.gradient_norm_ood import bench_gradient_norm_ood
from quant_fund.models.knn_ood import bench_knn_ood
from quant_fund.models.mahalanobis_ood import bench_mahalanobis_ood
from quant_fund.models.max_softmax_ood import bench_max_softmax_ood
from quant_fund.models.vim_ood import bench_vim_ood

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


def bench_mahalanobis_ood_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mahalanobis_ood", bench_mahalanobis_ood(seed=_SEED + 780)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mahalanobis_ood bench failed: {exc}") from exc


def bench_max_softmax_ood_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("max_softmax_ood", bench_max_softmax_ood(seed=_SEED + 781)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"max_softmax_ood bench failed: {exc}") from exc


def bench_gradient_norm_ood_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gradient_norm_ood", bench_gradient_norm_ood(seed=_SEED + 782)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gradient_norm_ood bench failed: {exc}") from exc


def bench_energy_ood_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("energy_ood", bench_energy_ood(seed=_SEED + 783)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"energy_ood bench failed: {exc}") from exc


def bench_knn_ood_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("knn_ood", bench_knn_ood(seed=_SEED + 784)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"knn_ood bench failed: {exc}") from exc


def bench_vim_ood_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vim_ood", bench_vim_ood(seed=_SEED + 785)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vim_ood bench failed: {exc}") from exc
