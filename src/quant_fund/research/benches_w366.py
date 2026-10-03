"""Wave-366 algebraic-topology-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cw_complex import bench_cw_complex
from quant_fund.models.excision import bench_excision
from quant_fund.models.homotopy_group import bench_homotopy_group
from quant_fund.models.poincare_dual import bench_poincare_dual
from quant_fund.models.singular_homology import bench_singular_homology
from quant_fund.models.spectral_seq_toy import bench_spectral_seq_toy

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


def bench_singular_homology_family(seed: int = _SEED + 2103) -> dict[str, float]:
    return _floats(_finite_blob("singular_homology", bench_singular_homology(seed)))


def bench_cw_complex_family(seed: int = _SEED + 2104) -> dict[str, float]:
    return _floats(_finite_blob("cw_complex", bench_cw_complex(seed)))


def bench_spectral_seq_toy_family(seed: int = _SEED + 2105) -> dict[str, float]:
    return _floats(_finite_blob("spectral_seq_toy", bench_spectral_seq_toy(seed)))


def bench_homotopy_group_family(seed: int = _SEED + 2106) -> dict[str, float]:
    return _floats(_finite_blob("homotopy_group", bench_homotopy_group(seed)))


def bench_excision_family(seed: int = _SEED + 2107) -> dict[str, float]:
    return _floats(_finite_blob("excision", bench_excision(seed)))


def bench_poincare_dual_family(seed: int = _SEED + 2108) -> dict[str, float]:
    return _floats(_finite_blob("poincare_dual", bench_poincare_dual(seed)))
