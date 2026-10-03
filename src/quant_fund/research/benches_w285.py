"""Wave-285 information-theory canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.blahut_arimoto import bench_blahut_arimoto
from quant_fund.models.elias_gamma import bench_elias_gamma
from quant_fund.models.kl_knn import bench_kl_knn
from quant_fund.models.markov_entropy import bench_markov_entropy
from quant_fund.models.miller_madow import bench_miller_madow
from quant_fund.models.type_class import bench_type_class

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


def bench_markov_entropy_family(seed: int = _SEED + 1616) -> dict[str, float]:
    return _floats(_finite_blob("markov_entropy", bench_markov_entropy(seed)))


def bench_blahut_arimoto_family(seed: int = _SEED + 1617) -> dict[str, float]:
    return _floats(_finite_blob("blahut_arimoto", bench_blahut_arimoto(seed)))


def bench_kl_knn_family(seed: int = _SEED + 1618) -> dict[str, float]:
    return _floats(_finite_blob("kl_knn", bench_kl_knn(seed)))


def bench_type_class_family(seed: int = _SEED + 1619) -> dict[str, float]:
    return _floats(_finite_blob("type_class", bench_type_class(seed)))


def bench_elias_gamma_family(seed: int = _SEED + 1620) -> dict[str, float]:
    return _floats(_finite_blob("elias_gamma", bench_elias_gamma(seed)))


def bench_miller_madow_family(seed: int = _SEED + 1621) -> dict[str, float]:
    return _floats(_finite_blob("miller_madow", bench_miller_madow(seed)))
