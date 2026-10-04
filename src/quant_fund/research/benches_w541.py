"""Wave-541 complex-analysis-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.jensen_formula import bench_jensen_formula
from quant_fund.models.montel_normal import bench_montel_normal
from quant_fund.models.picard_thm import bench_picard_thm
from quant_fund.models.riemann_mapping import bench_riemann_mapping
from quant_fund.models.runge_approx import bench_runge_approx
from quant_fund.models.schwarz_lemma import bench_schwarz_lemma

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


def bench_riemann_mapping_family(
    seed: int = _SEED + 3152,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "riemann_mapping",
            bench_riemann_mapping(seed),
        )
    )


def bench_schwarz_lemma_family(seed: int = _SEED + 3153) -> dict[str, float]:
    return _floats(_finite_blob("schwarz_lemma", bench_schwarz_lemma(seed)))


def bench_picard_thm_family(seed: int = _SEED + 3154) -> dict[str, float]:
    return _floats(_finite_blob("picard_thm", bench_picard_thm(seed)))


def bench_montel_normal_family(seed: int = _SEED + 3155) -> dict[str, float]:
    return _floats(_finite_blob("montel_normal", bench_montel_normal(seed)))


def bench_runge_approx_family(seed: int = _SEED + 3156) -> dict[str, float]:
    return _floats(_finite_blob("runge_approx", bench_runge_approx(seed)))


def bench_jensen_formula_family(
    seed: int = _SEED + 3157,
) -> dict[str, float]:
    return _floats(_finite_blob("jensen_formula", bench_jensen_formula(seed)))
