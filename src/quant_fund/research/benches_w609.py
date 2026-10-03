"""Wave-609 spectral-AG-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brave_new_ring import (
    bench_brave_new_ring,
)
from quant_fund.models.e_infty_space import (
    bench_e_infty_space,
)
from quant_fund.models.formal_moduli import (
    bench_formal_moduli,
)
from quant_fund.models.log_ring import bench_log_ring
from quant_fund.models.orient_cohom import bench_orient_cohom
from quant_fund.models.thom_constr import bench_thom_constr

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


def bench_e_infty_space_family(
    seed: int = _SEED + 3560,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "e_infty_space",
            bench_e_infty_space(seed),
        )
    )


def bench_brave_new_ring_family(
    seed: int = _SEED + 3561,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "brave_new_ring",
            bench_brave_new_ring(seed),
        )
    )


def bench_thom_constr_family(
    seed: int = _SEED + 3562,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thom_constr",
            bench_thom_constr(seed),
        )
    )


def bench_log_ring_family(seed: int = _SEED + 3563) -> dict[str, float]:
    return _floats(_finite_blob("log_ring", bench_log_ring(seed)))


def bench_orient_cohom_family(
    seed: int = _SEED + 3564,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "orient_cohom",
            bench_orient_cohom(seed),
        )
    )


def bench_formal_moduli_family(
    seed: int = _SEED + 3565,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "formal_moduli",
            bench_formal_moduli(seed),
        )
    )
