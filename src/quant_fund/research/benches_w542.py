"""Wave-542 several-complex-variables bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.d_bar_neumann import bench_d_bar_neumann
from quant_fund.models.domain_holo import bench_domain_holo
from quant_fund.models.hartogs_thm import bench_hartogs_thm
from quant_fund.models.levi_problem import bench_levi_problem
from quant_fund.models.oka_coherence import bench_oka_coherence
from quant_fund.models.pseudoconvex import bench_pseudoconvex

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


def bench_hartogs_thm_family(seed: int = _SEED + 3158) -> dict[str, float]:
    return _floats(_finite_blob("hartogs_thm", bench_hartogs_thm(seed)))


def bench_domain_holo_family(seed: int = _SEED + 3159) -> dict[str, float]:
    return _floats(_finite_blob("domain_holo", bench_domain_holo(seed)))


def bench_pseudoconvex_family(seed: int = _SEED + 3160) -> dict[str, float]:
    return _floats(_finite_blob("pseudoconvex", bench_pseudoconvex(seed)))


def bench_levi_problem_family(seed: int = _SEED + 3161) -> dict[str, float]:
    return _floats(_finite_blob("levi_problem", bench_levi_problem(seed)))


def bench_oka_coherence_family(seed: int = _SEED + 3162) -> dict[str, float]:
    return _floats(_finite_blob("oka_coherence", bench_oka_coherence(seed)))


def bench_d_bar_neumann_family(seed: int = _SEED + 3163) -> dict[str, float]:
    return _floats(_finite_blob("d_bar_neumann", bench_d_bar_neumann(seed)))
