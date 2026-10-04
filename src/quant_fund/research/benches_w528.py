"""Wave-528 thermodynamic-formalism bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.equilibrium_state import bench_equilibrium_state
from quant_fund.models.lasota_yorke import bench_lasota_yorke
from quant_fund.models.pressure_thm import bench_pressure_thm
from quant_fund.models.ruelle_zeta import bench_ruelle_zeta
from quant_fund.models.thermo_formal import bench_thermo_formal
from quant_fund.models.transfer_op import bench_transfer_op

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


def bench_transfer_op_family(seed: int = _SEED + 3074) -> dict[str, float]:
    return _floats(_finite_blob("transfer_op", bench_transfer_op(seed)))


def bench_thermo_formal_family(seed: int = _SEED + 3075) -> dict[str, float]:
    return _floats(_finite_blob("thermo_formal", bench_thermo_formal(seed)))


def bench_pressure_thm_family(seed: int = _SEED + 3076) -> dict[str, float]:
    return _floats(_finite_blob("pressure_thm", bench_pressure_thm(seed)))


def bench_equilibrium_state_family(
    seed: int = _SEED + 3077,
) -> dict[str, float]:
    return _floats(_finite_blob("equilibrium_state", bench_equilibrium_state(seed)))


def bench_ruelle_zeta_family(seed: int = _SEED + 3078) -> dict[str, float]:
    return _floats(_finite_blob("ruelle_zeta", bench_ruelle_zeta(seed)))


def bench_lasota_yorke_family(seed: int = _SEED + 3079) -> dict[str, float]:
    return _floats(_finite_blob("lasota_yorke", bench_lasota_yorke(seed)))
