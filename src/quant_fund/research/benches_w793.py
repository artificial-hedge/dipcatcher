"""Wave-793 stochastic-expansion bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cubature_wiener import (
    bench_cubature_wiener,
)
from quant_fund.models.milstein_scheme import (
    bench_milstein_scheme,
)
from quant_fund.models.rough_vol2 import bench_rough_vol2
from quant_fund.models.stochastic_taylor import (
    bench_stochastic_taylor,
)
from quant_fund.models.wagner_platen import (
    bench_wagner_platen,
)
from quant_fund.models.wong_zakai import (
    bench_wong_zakai,
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


def bench_wong_zakai_family(
    seed: int = _SEED + 18200,
) -> dict[str, float]:
    return _floats(_finite_blob("wong_zakai", bench_wong_zakai(seed)))


def bench_stochastic_taylor_family(
    seed: int = _SEED + 18201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_taylor",
            bench_stochastic_taylor(seed),
        )
    )


def bench_milstein_scheme_family(
    seed: int = _SEED + 18202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "milstein_scheme",
            bench_milstein_scheme(seed),
        )
    )


def bench_wagner_platen_family(
    seed: int = _SEED + 18203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wagner_platen",
            bench_wagner_platen(seed),
        )
    )


def bench_cubature_wiener_family(
    seed: int = _SEED + 18204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cubature_wiener",
            bench_cubature_wiener(seed),
        )
    )


def bench_rough_vol2_family(
    seed: int = _SEED + 18205,
) -> dict[str, float]:
    return _floats(_finite_blob("rough_vol2", bench_rough_vol2(seed)))
