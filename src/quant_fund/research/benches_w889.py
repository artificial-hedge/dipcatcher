"""Wave-889 FE-basis/sequence bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chebyshev_u import (
    bench_chebyshev_u,
)
from quant_fund.models.epsilon_algo import (
    bench_epsilon_algo,
)
from quant_fund.models.hexahedral_basis import (
    bench_hexahedral_basis,
)
from quant_fund.models.quadrilateral_basis import (
    bench_quadrilateral_basis,
)
from quant_fund.models.spline_theory import (
    bench_spline_theory,
)
from quant_fund.models.walsh_table import (
    bench_walsh_table,
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


def bench_quadrilateral_basis_family(
    seed: int = _SEED + 27700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quadrilateral_basis",
            bench_quadrilateral_basis(seed),
        )
    )


def bench_hexahedral_basis_family(
    seed: int = _SEED + 27701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hexahedral_basis",
            bench_hexahedral_basis(seed),
        )
    )


def bench_chebyshev_u_family(
    seed: int = _SEED + 27702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chebyshev_u",
            bench_chebyshev_u(seed),
        )
    )


def bench_walsh_table_family(
    seed: int = _SEED + 27703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "walsh_table",
            bench_walsh_table(seed),
        )
    )


def bench_epsilon_algo_family(
    seed: int = _SEED + 27704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "epsilon_algo",
            bench_epsilon_algo(seed),
        )
    )


def bench_spline_theory_family(
    seed: int = _SEED + 27705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spline_theory",
            bench_spline_theory(seed),
        )
    )
