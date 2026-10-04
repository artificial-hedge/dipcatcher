"""Wave-643 homotopy-18 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.calc_tower import bench_calc_tower
from quant_fund.models.goodwillie_deriv import (
    bench_goodwillie_deriv,
)
from quant_fund.models.kervaire_inv import (
    bench_kervaire_inv,
)
from quant_fund.models.mahowald_inv import (
    bench_mahowald_inv,
)
from quant_fund.models.snaith_split import (
    bench_snaith_split,
)
from quant_fund.models.toda_smith import bench_toda_smith

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


def bench_toda_smith_family(
    seed: int = _SEED + 3764,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "toda_smith",
            bench_toda_smith(seed),
        )
    )


def bench_mahowald_inv_family(
    seed: int = _SEED + 3765,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mahowald_inv",
            bench_mahowald_inv(seed),
        )
    )


def bench_calc_tower_family(
    seed: int = _SEED + 3766,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "calc_tower",
            bench_calc_tower(seed),
        )
    )


def bench_goodwillie_deriv_family(
    seed: int = _SEED + 3767,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "goodwillie_deriv",
            bench_goodwillie_deriv(seed),
        )
    )


def bench_snaith_split_family(
    seed: int = _SEED + 3768,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "snaith_split",
            bench_snaith_split(seed),
        )
    )


def bench_kervaire_inv_family(
    seed: int = _SEED + 3769,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kervaire_inv",
            bench_kervaire_inv(seed),
        )
    )
