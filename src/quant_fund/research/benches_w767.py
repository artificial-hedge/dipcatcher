"""Wave-767 Gaussian-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.borell_tis import bench_borell_tis
from quant_fund.models.fernique_thm import bench_fernique_thm
from quant_fund.models.gordon_thm import bench_gordon_thm
from quant_fund.models.slepian_lemma import bench_slepian_lemma
from quant_fund.models.sudakov_min import bench_sudakov_min
from quant_fund.models.talagrand_conc import (
    bench_talagrand_conc,
)

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


def bench_slepian_lemma_family(
    seed: int = _SEED + 15600,
) -> dict[str, float]:
    return _floats(_finite_blob("slepian_lemma", bench_slepian_lemma(seed)))


def bench_fernique_thm_family(
    seed: int = _SEED + 15601,
) -> dict[str, float]:
    return _floats(_finite_blob("fernique_thm", bench_fernique_thm(seed)))


def bench_borell_tis_family(
    seed: int = _SEED + 15602,
) -> dict[str, float]:
    return _floats(_finite_blob("borell_tis", bench_borell_tis(seed)))


def bench_sudakov_min_family(
    seed: int = _SEED + 15603,
) -> dict[str, float]:
    return _floats(_finite_blob("sudakov_min", bench_sudakov_min(seed)))


def bench_talagrand_conc_family(
    seed: int = _SEED + 15604,
) -> dict[str, float]:
    return _floats(_finite_blob("talagrand_conc", bench_talagrand_conc(seed)))


def bench_gordon_thm_family(
    seed: int = _SEED + 15605,
) -> dict[str, float]:
    return _floats(_finite_blob("gordon_thm", bench_gordon_thm(seed)))
