"""Wave-891 optimization/IGA bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bfgs_update import (
    bench_bfgs_update,
)
from quant_fund.models.iga_colloc import (
    bench_iga_colloc,
)
from quant_fund.models.lebesgue_const import (
    bench_lebesgue_const,
)
from quant_fund.models.newton_armijo import (
    bench_newton_armijo,
)
from quant_fund.models.trimmed_cad import (
    bench_trimmed_cad,
)
from quant_fund.models.trust_region_dogleg import (
    bench_trust_region_dogleg,
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


def bench_trust_region_dogleg_family(
    seed: int = _SEED + 27900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "trust_region_dogleg",
            bench_trust_region_dogleg(seed),
        )
    )


def bench_bfgs_update_family(
    seed: int = _SEED + 27901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bfgs_update",
            bench_bfgs_update(seed),
        )
    )


def bench_lebesgue_const_family(
    seed: int = _SEED + 27902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lebesgue_const",
            bench_lebesgue_const(seed),
        )
    )


def bench_iga_colloc_family(
    seed: int = _SEED + 27903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "iga_colloc",
            bench_iga_colloc(seed),
        )
    )


def bench_trimmed_cad_family(
    seed: int = _SEED + 27904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "trimmed_cad",
            bench_trimmed_cad(seed),
        )
    )


def bench_newton_armijo_family(
    seed: int = _SEED + 27905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "newton_armijo",
            bench_newton_armijo(seed),
        )
    )
