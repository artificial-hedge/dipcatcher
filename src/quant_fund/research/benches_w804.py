"""Wave-804 Malliavin bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.absolute_cont import (
    bench_absolute_cont,
)
from quant_fund.models.density_bound import (
    bench_density_bound,
)
from quant_fund.models.malliavin_cov import (
    bench_malliavin_cov,
)
from quant_fund.models.nualart_zakai import (
    bench_nualart_zakai,
)
from quant_fund.models.smoothness_h import (
    bench_smoothness_h,
)
from quant_fund.models.watanabe_map import (
    bench_watanabe_map,
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


def bench_nualart_zakai_family(
    seed: int = _SEED + 19300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nualart_zakai",
            bench_nualart_zakai(seed),
        )
    )


def bench_watanabe_map_family(
    seed: int = _SEED + 19301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "watanabe_map",
            bench_watanabe_map(seed),
        )
    )


def bench_malliavin_cov_family(
    seed: int = _SEED + 19302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "malliavin_cov",
            bench_malliavin_cov(seed),
        )
    )


def bench_density_bound_family(
    seed: int = _SEED + 19303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "density_bound",
            bench_density_bound(seed),
        )
    )


def bench_absolute_cont_family(
    seed: int = _SEED + 19304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "absolute_cont",
            bench_absolute_cont(seed),
        )
    )


def bench_smoothness_h_family(
    seed: int = _SEED + 19305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "smoothness_h",
            bench_smoothness_h(seed),
        )
    )
