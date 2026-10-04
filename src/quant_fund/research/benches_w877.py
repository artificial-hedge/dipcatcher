"""Wave-877 transport/SPn bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.discrete_ordinates import (
    bench_discrete_ordinates,
)
from quant_fund.models.moc_transport import (
    bench_moc_transport,
)
from quant_fund.models.pn_closure import (
    bench_pn_closure,
)
from quant_fund.models.spherical_harmonics import (
    bench_spherical_harmonics,
)
from quant_fund.models.spn_equations import (
    bench_spn_equations,
)
from quant_fund.models.transport_sn import (
    bench_transport_sn,
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


def bench_transport_sn_family(
    seed: int = _SEED + 26500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "transport_sn",
            bench_transport_sn(seed),
        )
    )


def bench_discrete_ordinates_family(
    seed: int = _SEED + 26501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "discrete_ordinates",
            bench_discrete_ordinates(seed),
        )
    )


def bench_spherical_harmonics_family(
    seed: int = _SEED + 26502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spherical_harmonics",
            bench_spherical_harmonics(seed),
        )
    )


def bench_spn_equations_family(
    seed: int = _SEED + 26503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spn_equations",
            bench_spn_equations(seed),
        )
    )


def bench_moc_transport_family(
    seed: int = _SEED + 26504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "moc_transport",
            bench_moc_transport(seed),
        )
    )


def bench_pn_closure_family(
    seed: int = _SEED + 26505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pn_closure",
            bench_pn_closure(seed),
        )
    )
