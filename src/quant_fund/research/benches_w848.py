"""Wave-848 finite-element bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dof_management import (
    bench_dof_management,
)
from quant_fund.models.edge_elements import (
    bench_edge_elements,
)
from quant_fund.models.fem_assembly import (
    bench_fem_assembly,
)
from quant_fund.models.isoparametric_map import (
    bench_isoparametric_map,
)
from quant_fund.models.quadrature_rules import (
    bench_quadrature_rules,
)
from quant_fund.models.triangular_basis import (
    bench_triangular_basis,
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


def bench_fem_assembly_family(
    seed: int = _SEED + 23600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fem_assembly",
            bench_fem_assembly(seed),
        )
    )


def bench_isoparametric_map_family(
    seed: int = _SEED + 23601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "isoparametric_map",
            bench_isoparametric_map(seed),
        )
    )


def bench_quadrature_rules_family(
    seed: int = _SEED + 23602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quadrature_rules",
            bench_quadrature_rules(seed),
        )
    )


def bench_triangular_basis_family(
    seed: int = _SEED + 23603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "triangular_basis",
            bench_triangular_basis(seed),
        )
    )


def bench_edge_elements_family(
    seed: int = _SEED + 23604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "edge_elements",
            bench_edge_elements(seed),
        )
    )


def bench_dof_management_family(
    seed: int = _SEED + 23605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dof_management",
            bench_dof_management(seed),
        )
    )
