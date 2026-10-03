"""Wave-849 finite-volume bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.compact_scheme import (
    bench_compact_scheme,
)
from quant_fund.models.crank_nicholson2 import (
    bench_crank_nicholson2,
)
from quant_fund.models.fdm_grid import (
    bench_fdm_grid,
)
from quant_fund.models.flux_splitting import (
    bench_flux_splitting,
)
from quant_fund.models.muscl_reconstruct import (
    bench_muscl_reconstruct,
)
from quant_fund.models.upwind_scheme import (
    bench_upwind_scheme,
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


def bench_fdm_grid_family(
    seed: int = _SEED + 23700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fdm_grid",
            bench_fdm_grid(seed),
        )
    )


def bench_compact_scheme_family(
    seed: int = _SEED + 23701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "compact_scheme",
            bench_compact_scheme(seed),
        )
    )


def bench_crank_nicholson2_family(
    seed: int = _SEED + 23702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "crank_nicholson2",
            bench_crank_nicholson2(seed),
        )
    )


def bench_upwind_scheme_family(
    seed: int = _SEED + 23703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "upwind_scheme",
            bench_upwind_scheme(seed),
        )
    )


def bench_muscl_reconstruct_family(
    seed: int = _SEED + 23704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "muscl_reconstruct",
            bench_muscl_reconstruct(seed),
        )
    )


def bench_flux_splitting_family(
    seed: int = _SEED + 23705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "flux_splitting",
            bench_flux_splitting(seed),
        )
    )
