"""Wave-885 DG-flux/asymptotics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.averaging_method import (
    bench_averaging_method,
)
from quant_fund.models.entropy_stable_dg import (
    bench_entropy_stable_dg,
)
from quant_fund.models.hyperasymptotic import (
    bench_hyperasymptotic,
)
from quant_fund.models.laplace_method import (
    bench_laplace_method,
)
from quant_fund.models.ldg_flux import (
    bench_ldg_flux,
)
from quant_fund.models.wkb_turning import (
    bench_wkb_turning,
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


def bench_ldg_flux_family(
    seed: int = _SEED + 27300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ldg_flux",
            bench_ldg_flux(seed),
        )
    )


def bench_entropy_stable_dg_family(
    seed: int = _SEED + 27301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "entropy_stable_dg",
            bench_entropy_stable_dg(seed),
        )
    )


def bench_wkb_turning_family(
    seed: int = _SEED + 27302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wkb_turning",
            bench_wkb_turning(seed),
        )
    )


def bench_averaging_method_family(
    seed: int = _SEED + 27303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "averaging_method",
            bench_averaging_method(seed),
        )
    )


def bench_laplace_method_family(
    seed: int = _SEED + 27304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "laplace_method",
            bench_laplace_method(seed),
        )
    )


def bench_hyperasymptotic_family(
    seed: int = _SEED + 27305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hyperasymptotic",
            bench_hyperasymptotic(seed),
        )
    )
