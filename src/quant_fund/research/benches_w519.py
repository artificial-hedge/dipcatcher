"""Wave-519 modular-forms bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cusp_form import bench_cusp_form
from quant_fund.models.dedekind_eta import bench_dedekind_eta
from quant_fund.models.eisenstein_srs2 import bench_eisenstein_srs2
from quant_fund.models.hecke_op2 import bench_hecke_op2
from quant_fund.models.modular_form import bench_modular_form
from quant_fund.models.theta_func import bench_theta_func

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


def bench_modular_form_family(seed: int = _SEED + 3020) -> dict[str, float]:
    return _floats(_finite_blob("modular_form", bench_modular_form(seed)))


def bench_hecke_op2_family(seed: int = _SEED + 3021) -> dict[str, float]:
    return _floats(_finite_blob("hecke_op2", bench_hecke_op2(seed)))


def bench_eisenstein_srs2_family(seed: int = _SEED + 3022) -> dict[str, float]:
    return _floats(_finite_blob("eisenstein_srs2", bench_eisenstein_srs2(seed)))


def bench_cusp_form_family(seed: int = _SEED + 3023) -> dict[str, float]:
    return _floats(_finite_blob("cusp_form", bench_cusp_form(seed)))


def bench_theta_func_family(seed: int = _SEED + 3024) -> dict[str, float]:
    return _floats(_finite_blob("theta_func", bench_theta_func(seed)))


def bench_dedekind_eta_family(seed: int = _SEED + 3025) -> dict[str, float]:
    return _floats(_finite_blob("dedekind_eta", bench_dedekind_eta(seed)))
