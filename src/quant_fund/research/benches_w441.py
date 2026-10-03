"""Wave-441 Galois-representations bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.filtered_module import bench_filtered_module
from quant_fund.models.fontaine_ring import bench_fontaine_ring
from quant_fund.models.gal_rep import bench_gal_rep
from quant_fund.models.hecke_eigensys import bench_hecke_eigensys
from quant_fund.models.ribet_toy import bench_ribet_toy
from quant_fund.models.weil_deligne import bench_weil_deligne

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


def bench_gal_rep_family(
    seed: int = _SEED + 2552,
) -> dict[str, float]:
    return _floats(_finite_blob("gal_rep", bench_gal_rep(seed)))


def bench_fontaine_ring_family(
    seed: int = _SEED + 2553,
) -> dict[str, float]:
    return _floats(_finite_blob("fontaine_ring", bench_fontaine_ring(seed)))


def bench_filtered_module_family(
    seed: int = _SEED + 2554,
) -> dict[str, float]:
    return _floats(_finite_blob("filtered_module", bench_filtered_module(seed)))


def bench_weil_deligne_family(
    seed: int = _SEED + 2555,
) -> dict[str, float]:
    return _floats(_finite_blob("weil_deligne", bench_weil_deligne(seed)))


def bench_hecke_eigensys_family(
    seed: int = _SEED + 2556,
) -> dict[str, float]:
    return _floats(_finite_blob("hecke_eigensys", bench_hecke_eigensys(seed)))


def bench_ribet_toy_family(
    seed: int = _SEED + 2557,
) -> dict[str, float]:
    return _floats(_finite_blob("ribet_toy", bench_ribet_toy(seed)))
