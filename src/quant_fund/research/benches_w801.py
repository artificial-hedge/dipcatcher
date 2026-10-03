"""Wave-801 rough-volatility bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.fractional_heston import (
    bench_fractional_heston,
)
from quant_fund.models.multifactor_rough import (
    bench_multifactor_rough,
)
from quant_fund.models.rough_bergomi import (
    bench_rough_bergomi,
)
from quant_fund.models.rough_sabr import (
    bench_rough_sabr,
)
from quant_fund.models.rough_variance import (
    bench_rough_variance,
)
from quant_fund.models.volterra_sde import (
    bench_volterra_sde,
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


def bench_fractional_heston_family(
    seed: int = _SEED + 19000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fractional_heston",
            bench_fractional_heston(seed),
        )
    )


def bench_rough_bergomi_family(
    seed: int = _SEED + 19001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rough_bergomi",
            bench_rough_bergomi(seed),
        )
    )


def bench_rough_sabr_family(
    seed: int = _SEED + 19002,
) -> dict[str, float]:
    return _floats(_finite_blob("rough_sabr", bench_rough_sabr(seed)))


def bench_rough_variance_family(
    seed: int = _SEED + 19003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rough_variance",
            bench_rough_variance(seed),
        )
    )


def bench_volterra_sde_family(
    seed: int = _SEED + 19004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "volterra_sde",
            bench_volterra_sde(seed),
        )
    )


def bench_multifactor_rough_family(
    seed: int = _SEED + 19005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "multifactor_rough",
            bench_multifactor_rough(seed),
        )
    )
