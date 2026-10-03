"""Wave-372 stochastic-analysis canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.girsanov import bench_girsanov
from quant_fund.models.ito_lemma import bench_ito_lemma
from quant_fund.models.local_time import bench_local_time
from quant_fund.models.malliavin import bench_malliavin
from quant_fund.models.quadratic_var import bench_quadratic_var
from quant_fund.models.sde_strong import bench_sde_strong

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


def bench_ito_lemma_family(seed: int = _SEED + 2138) -> dict[str, float]:
    return _floats(_finite_blob("ito_lemma", bench_ito_lemma(seed)))


def bench_girsanov_family(seed: int = _SEED + 2139) -> dict[str, float]:
    return _floats(_finite_blob("girsanov", bench_girsanov(seed)))


def bench_sde_strong_family(seed: int = _SEED + 2140) -> dict[str, float]:
    return _floats(_finite_blob("sde_strong", bench_sde_strong(seed)))


def bench_local_time_family(seed: int = _SEED + 2141) -> dict[str, float]:
    return _floats(_finite_blob("local_time", bench_local_time(seed)))


def bench_quadratic_var_family(seed: int = _SEED + 2142) -> dict[str, float]:
    return _floats(_finite_blob("quadratic_var", bench_quadratic_var(seed)))


def bench_malliavin_family(seed: int = _SEED + 2143) -> dict[str, float]:
    return _floats(_finite_blob("malliavin", bench_malliavin(seed)))
