"""Wave-335 category-2/topos canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adjoint_check import bench_adjoint_check
from quant_fund.models.cat_colimit import bench_cat_colimit
from quant_fund.models.exponential_obj import bench_exponential_obj
from quant_fund.models.fin_limit import bench_fin_limit
from quant_fund.models.subobject_classifier import bench_subobject_classifier
from quant_fund.models.yoneda_embed import bench_yoneda_embed

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_fin_limit_family(seed: int = _SEED + 1917) -> dict[str, float]:
    return _floats(_finite_blob("fin_limit", bench_fin_limit(seed)))


def bench_subobject_classifier_family(seed: int = _SEED + 1918) -> dict[str, float]:
    return _floats(_finite_blob("subobject_classifier", bench_subobject_classifier(seed)))


def bench_exponential_obj_family(seed: int = _SEED + 1919) -> dict[str, float]:
    return _floats(_finite_blob("exponential_obj", bench_exponential_obj(seed)))


def bench_yoneda_embed_family(seed: int = _SEED + 1920) -> dict[str, float]:
    return _floats(_finite_blob("yoneda_embed", bench_yoneda_embed(seed)))


def bench_adjoint_check_family(seed: int = _SEED + 1921) -> dict[str, float]:
    return _floats(_finite_blob("adjoint_check", bench_adjoint_check(seed)))


def bench_cat_colimit_family(seed: int = _SEED + 1922) -> dict[str, float]:
    return _floats(_finite_blob("cat_colimit", bench_cat_colimit(seed)))
