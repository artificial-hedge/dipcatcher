"""Wave-568 symplectic-field-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.contact_homology3 import (
    bench_contact_homology3,
)
from quant_fund.models.eliashberg_givental import (
    bench_eliashberg_givental,
)
from quant_fund.models.floer_homol import bench_floer_homol
from quant_fund.models.reeb_orbit import bench_reeb_orbit
from quant_fund.models.sft_algebra import bench_sft_algebra
from quant_fund.models.symplectic_field import (
    bench_symplectic_field,
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


def bench_symplectic_field_family(
    seed: int = _SEED + 3314,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "symplectic_field",
            bench_symplectic_field(seed),
        )
    )


def bench_contact_homology3_family(
    seed: int = _SEED + 3315,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "contact_homology3",
            bench_contact_homology3(seed),
        )
    )


def bench_floer_homol_family(seed: int = _SEED + 3316) -> dict[str, float]:
    return _floats(_finite_blob("floer_homol", bench_floer_homol(seed)))


def bench_reeb_orbit_family(seed: int = _SEED + 3317) -> dict[str, float]:
    return _floats(_finite_blob("reeb_orbit", bench_reeb_orbit(seed)))


def bench_sft_algebra_family(seed: int = _SEED + 3318) -> dict[str, float]:
    return _floats(_finite_blob("sft_algebra", bench_sft_algebra(seed)))


def bench_eliashberg_givental_family(
    seed: int = _SEED + 3319,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "eliashberg_givental",
            bench_eliashberg_givental(seed),
        )
    )
