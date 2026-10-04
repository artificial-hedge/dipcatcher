"""Wave-353 ODE-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gronwall_lemma import bench_gronwall_lemma
from quant_fund.models.lyapunov_stability import bench_lyapunov_stability
from quant_fund.models.phase_plane import bench_phase_plane
from quant_fund.models.picard_lindelof import bench_picard_lindelof
from quant_fund.models.sturm_liouville import bench_sturm_liouville
from quant_fund.models.variation_params import bench_variation_params

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


def bench_picard_lindelof_family(seed: int = _SEED + 2025) -> dict[str, float]:
    return _floats(_finite_blob("picard_lindelof", bench_picard_lindelof(seed)))


def bench_gronwall_lemma_family(seed: int = _SEED + 2026) -> dict[str, float]:
    return _floats(_finite_blob("gronwall_lemma", bench_gronwall_lemma(seed)))


def bench_sturm_liouville_family(seed: int = _SEED + 2027) -> dict[str, float]:
    return _floats(_finite_blob("sturm_liouville", bench_sturm_liouville(seed)))


def bench_phase_plane_family(seed: int = _SEED + 2028) -> dict[str, float]:
    return _floats(_finite_blob("phase_plane", bench_phase_plane(seed)))


def bench_lyapunov_stability_family(seed: int = _SEED + 2029) -> dict[str, float]:
    return _floats(_finite_blob("lyapunov_stability", bench_lyapunov_stability(seed)))


def bench_variation_params_family(seed: int = _SEED + 2030) -> dict[str, float]:
    return _floats(_finite_blob("variation_params", bench_variation_params(seed)))
