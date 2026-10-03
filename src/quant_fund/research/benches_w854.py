"""Wave-854 radial-basis-function bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gaussian_rbf import (
    bench_gaussian_rbf,
)
from quant_fund.models.kansa_collocation import (
    bench_kansa_collocation,
)
from quant_fund.models.multiquadric_rbf import (
    bench_multiquadric_rbf,
)
from quant_fund.models.rbf_finite_diff import (
    bench_rbf_finite_diff,
)
from quant_fund.models.rbf_interp import (
    bench_rbf_interp,
)
from quant_fund.models.wendland_rbf import (
    bench_wendland_rbf,
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


def bench_rbf_interp_family(
    seed: int = _SEED + 24200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rbf_interp",
            bench_rbf_interp(seed),
        )
    )


def bench_gaussian_rbf_family(
    seed: int = _SEED + 24201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gaussian_rbf",
            bench_gaussian_rbf(seed),
        )
    )


def bench_multiquadric_rbf_family(
    seed: int = _SEED + 24202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "multiquadric_rbf",
            bench_multiquadric_rbf(seed),
        )
    )


def bench_kansa_collocation_family(
    seed: int = _SEED + 24203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kansa_collocation",
            bench_kansa_collocation(seed),
        )
    )


def bench_rbf_finite_diff_family(
    seed: int = _SEED + 24204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rbf_finite_diff",
            bench_rbf_finite_diff(seed),
        )
    )


def bench_wendland_rbf_family(
    seed: int = _SEED + 24205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wendland_rbf",
            bench_wendland_rbf(seed),
        )
    )
