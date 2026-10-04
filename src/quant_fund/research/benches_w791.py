"""Wave-791 SPDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.doering_mueller import (
    bench_doering_mueller,
)
from quant_fund.models.kpz_equation import (
    bench_kpz_equation,
)
from quant_fund.models.paracontrolled_spde import (
    bench_paracontrolled_spde,
)
from quant_fund.models.quasilinear_spde import (
    bench_quasilinear_spde,
)
from quant_fund.models.spde_heat import bench_spde_heat
from quant_fund.models.stochastic_burgers import (
    bench_stochastic_burgers,
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


def bench_spde_heat_family(
    seed: int = _SEED + 18000,
) -> dict[str, float]:
    return _floats(_finite_blob("spde_heat", bench_spde_heat(seed)))


def bench_stochastic_burgers_family(
    seed: int = _SEED + 18001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_burgers",
            bench_stochastic_burgers(seed),
        )
    )


def bench_kpz_equation_family(
    seed: int = _SEED + 18002,
) -> dict[str, float]:
    return _floats(_finite_blob("kpz_equation", bench_kpz_equation(seed)))


def bench_doering_mueller_family(
    seed: int = _SEED + 18003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "doering_mueller",
            bench_doering_mueller(seed),
        )
    )


def bench_quasilinear_spde_family(
    seed: int = _SEED + 18004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quasilinear_spde",
            bench_quasilinear_spde(seed),
        )
    )


def bench_paracontrolled_spde_family(
    seed: int = _SEED + 18005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "paracontrolled_spde",
            bench_paracontrolled_spde(seed),
        )
    )
