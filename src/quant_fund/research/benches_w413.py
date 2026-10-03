"""Wave-413 Galois-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abelian_ext import bench_abelian_ext
from quant_fund.models.artin_lemma import bench_artin_lemma
from quant_fund.models.frobenius_el import bench_frobenius_el
from quant_fund.models.inseparable import bench_inseparable
from quant_fund.models.kummer_ext import bench_kummer_ext
from quant_fund.models.normal_basis import bench_normal_basis

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


def bench_artin_lemma_family(
    seed: int = _SEED + 2384,
) -> dict[str, float]:
    return _floats(_finite_blob("artin_lemma", bench_artin_lemma(seed)))


def bench_normal_basis_family(
    seed: int = _SEED + 2385,
) -> dict[str, float]:
    return _floats(_finite_blob("normal_basis", bench_normal_basis(seed)))


def bench_kummer_ext_family(
    seed: int = _SEED + 2386,
) -> dict[str, float]:
    return _floats(_finite_blob("kummer_ext", bench_kummer_ext(seed)))


def bench_abelian_ext_family(
    seed: int = _SEED + 2387,
) -> dict[str, float]:
    return _floats(_finite_blob("abelian_ext", bench_abelian_ext(seed)))


def bench_frobenius_el_family(
    seed: int = _SEED + 2388,
) -> dict[str, float]:
    return _floats(_finite_blob("frobenius_el", bench_frobenius_el(seed)))


def bench_inseparable_family(
    seed: int = _SEED + 2389,
) -> dict[str, float]:
    return _floats(_finite_blob("inseparable", bench_inseparable(seed)))
