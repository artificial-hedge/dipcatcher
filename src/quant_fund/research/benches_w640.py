"""Wave-640 spectral-AG-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.azure_space import (
    bench_azure_space,
)
from quant_fund.models.elliptic_cohom2 import (
    bench_elliptic_cohom2,
)
from quant_fund.models.spectral_etale import (
    bench_spectral_etale,
)
from quant_fund.models.spectral_group import (
    bench_spectral_group,
)
from quant_fund.models.spectral_scheme3 import (
    bench_spectral_scheme3,
)
from quant_fund.models.spectral_smooth import (
    bench_spectral_smooth,
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


def bench_spectral_group_family(
    seed: int = _SEED + 3746,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_group",
            bench_spectral_group(seed),
        )
    )


def bench_azure_space_family(
    seed: int = _SEED + 3747,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "azure_space",
            bench_azure_space(seed),
        )
    )


def bench_spectral_scheme3_family(
    seed: int = _SEED + 3748,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_scheme3",
            bench_spectral_scheme3(seed),
        )
    )


def bench_spectral_smooth_family(
    seed: int = _SEED + 3749,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_smooth",
            bench_spectral_smooth(seed),
        )
    )


def bench_spectral_etale_family(
    seed: int = _SEED + 3750,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_etale",
            bench_spectral_etale(seed),
        )
    )


def bench_elliptic_cohom2_family(
    seed: int = _SEED + 3751,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "elliptic_cohom2",
            bench_elliptic_cohom2(seed),
        )
    )
