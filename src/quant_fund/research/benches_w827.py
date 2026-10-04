"""Wave-827 random-series bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chung_series import (
    bench_chung_series,
)
from quant_fund.models.ito_nisio import (
    bench_ito_nisio,
)
from quant_fund.models.kolmogorov_two import (
    bench_kolmogorov_two,
)
from quant_fund.models.ortega_series import (
    bench_ortega_series,
)
from quant_fund.models.salem_zygmund import (
    bench_salem_zygmund,
)
from quant_fund.models.three_series import (
    bench_three_series,
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


def bench_three_series_family(
    seed: int = _SEED + 21500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "three_series",
            bench_three_series(seed),
        )
    )


def bench_kolmogorov_two_family(
    seed: int = _SEED + 21501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kolmogorov_two",
            bench_kolmogorov_two(seed),
        )
    )


def bench_ito_nisio_family(
    seed: int = _SEED + 21502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ito_nisio",
            bench_ito_nisio(seed),
        )
    )


def bench_chung_series_family(
    seed: int = _SEED + 21503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chung_series",
            bench_chung_series(seed),
        )
    )


def bench_ortega_series_family(
    seed: int = _SEED + 21504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ortega_series",
            bench_ortega_series(seed),
        )
    )


def bench_salem_zygmund_family(
    seed: int = _SEED + 21505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "salem_zygmund",
            bench_salem_zygmund(seed),
        )
    )
