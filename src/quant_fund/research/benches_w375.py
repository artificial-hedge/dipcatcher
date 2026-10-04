"""Wave-375 algebraic-geometry-5 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.blowup import bench_blowup
from quant_fund.models.elliptic_group import bench_elliptic_group
from quant_fund.models.moduli_stable import bench_moduli_stable
from quant_fund.models.riemann_roch import bench_riemann_roch
from quant_fund.models.scheme_local import bench_scheme_local
from quant_fund.models.sheaf_cohomology import bench_sheaf_cohomology

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


def bench_riemann_roch_family(seed: int = _SEED + 2156) -> dict[str, float]:
    return _floats(_finite_blob("riemann_roch", bench_riemann_roch(seed)))


def bench_sheaf_cohomology_family(seed: int = _SEED + 2157) -> dict[str, float]:
    return _floats(_finite_blob("sheaf_cohomology", bench_sheaf_cohomology(seed)))


def bench_scheme_local_family(seed: int = _SEED + 2158) -> dict[str, float]:
    return _floats(_finite_blob("scheme_local", bench_scheme_local(seed)))


def bench_blowup_family(seed: int = _SEED + 2159) -> dict[str, float]:
    return _floats(_finite_blob("blowup", bench_blowup(seed)))


def bench_elliptic_group_family(seed: int = _SEED + 2160) -> dict[str, float]:
    return _floats(_finite_blob("elliptic_group", bench_elliptic_group(seed)))


def bench_moduli_stable_family(seed: int = _SEED + 2161) -> dict[str, float]:
    return _floats(_finite_blob("moduli_stable", bench_moduli_stable(seed)))
