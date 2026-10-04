"""Wave-637 p-adic-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cocartesian_diamond import (
    bench_cocartesian_diamond,
)
from quant_fund.models.curve_padic import (
    bench_curve_padic,
)
from quant_fund.models.diamond_mod import (
    bench_diamond_mod,
)
from quant_fund.models.etale_phiphi import (
    bench_etale_phiphi,
)
from quant_fund.models.fargues_scholze2 import (
    bench_fargues_scholze2,
)
from quant_fund.models.scholze_bc import bench_scholze_bc

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


def bench_fargues_scholze2_family(
    seed: int = _SEED + 3728,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fargues_scholze2",
            bench_fargues_scholze2(seed),
        )
    )


def bench_curve_padic_family(
    seed: int = _SEED + 3729,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "curve_padic",
            bench_curve_padic(seed),
        )
    )


def bench_diamond_mod_family(
    seed: int = _SEED + 3730,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "diamond_mod",
            bench_diamond_mod(seed),
        )
    )


def bench_etale_phiphi_family(
    seed: int = _SEED + 3731,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etale_phiphi",
            bench_etale_phiphi(seed),
        )
    )


def bench_cocartesian_diamond_family(
    seed: int = _SEED + 3732,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cocartesian_diamond",
            bench_cocartesian_diamond(seed),
        )
    )


def bench_scholze_bc_family(
    seed: int = _SEED + 3733,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "scholze_bc",
            bench_scholze_bc(seed),
        )
    )
