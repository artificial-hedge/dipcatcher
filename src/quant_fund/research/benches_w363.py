"""Wave-363 real-analysis canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.baire_category import bench_baire_category
from quant_fund.models.cantor_set import bench_cantor_set
from quant_fund.models.egorov_thm import bench_egorov_thm
from quant_fund.models.fatou_lemma import bench_fatou_lemma
from quant_fund.models.monotone_conv import bench_monotone_conv
from quant_fund.models.vitali_set import bench_vitali_set

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


def bench_cantor_set_family(seed: int = _SEED + 2085) -> dict[str, float]:
    return _floats(_finite_blob("cantor_set", bench_cantor_set(seed)))


def bench_baire_category_family(seed: int = _SEED + 2086) -> dict[str, float]:
    return _floats(_finite_blob("baire_category", bench_baire_category(seed)))


def bench_vitali_set_family(seed: int = _SEED + 2087) -> dict[str, float]:
    return _floats(_finite_blob("vitali_set", bench_vitali_set(seed)))


def bench_egorov_thm_family(seed: int = _SEED + 2088) -> dict[str, float]:
    return _floats(_finite_blob("egorov_thm", bench_egorov_thm(seed)))


def bench_fatou_lemma_family(seed: int = _SEED + 2089) -> dict[str, float]:
    return _floats(_finite_blob("fatou_lemma", bench_fatou_lemma(seed)))


def bench_monotone_conv_family(seed: int = _SEED + 2090) -> dict[str, float]:
    return _floats(_finite_blob("monotone_conv", bench_monotone_conv(seed)))
