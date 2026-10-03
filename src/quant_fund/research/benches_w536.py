"""Wave-536 symplectic-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.contact_geom import bench_contact_geom
from quant_fund.models.gromov_nonsq import bench_gromov_nonsq
from quant_fund.models.hamiltonian_flow import bench_hamiltonian_flow
from quant_fund.models.lagrangian_mfd import bench_lagrangian_mfd
from quant_fund.models.poisson_bracket import bench_poisson_bracket
from quant_fund.models.symplectic_form import bench_symplectic_form

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


def bench_symplectic_form_family(
    seed: int = _SEED + 3122,
) -> dict[str, float]:
    return _floats(_finite_blob("symplectic_form", bench_symplectic_form(seed)))


def bench_lagrangian_mfd_family(seed: int = _SEED + 3123) -> dict[str, float]:
    return _floats(_finite_blob("lagrangian_mfd", bench_lagrangian_mfd(seed)))


def bench_hamiltonian_flow_family(
    seed: int = _SEED + 3124,
) -> dict[str, float]:
    return _floats(_finite_blob("hamiltonian_flow", bench_hamiltonian_flow(seed)))


def bench_poisson_bracket_family(
    seed: int = _SEED + 3125,
) -> dict[str, float]:
    return _floats(_finite_blob("poisson_bracket", bench_poisson_bracket(seed)))


def bench_contact_geom_family(seed: int = _SEED + 3126) -> dict[str, float]:
    return _floats(_finite_blob("contact_geom", bench_contact_geom(seed)))


def bench_gromov_nonsq_family(seed: int = _SEED + 3127) -> dict[str, float]:
    return _floats(_finite_blob("gromov_nonsq", bench_gromov_nonsq(seed)))
