"""Wave-831 empirical-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.covering_number import (
    bench_covering_number,
)
from quant_fund.models.entropy_integral import (
    bench_entropy_integral,
)
from quant_fund.models.metric_entropy import (
    bench_metric_entropy,
)
from quant_fund.models.rademacher_cplx import (
    bench_rademacher_cplx,
)
from quant_fund.models.symmetrization import (
    bench_symmetrization,
)
from quant_fund.models.uniform_clt import (
    bench_uniform_clt,
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


def bench_entropy_integral_family(
    seed: int = _SEED + 21900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "entropy_integral",
            bench_entropy_integral(seed),
        )
    )


def bench_uniform_clt_family(
    seed: int = _SEED + 21901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "uniform_clt",
            bench_uniform_clt(seed),
        )
    )


def bench_symmetrization_family(
    seed: int = _SEED + 21902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "symmetrization",
            bench_symmetrization(seed),
        )
    )


def bench_rademacher_cplx_family(
    seed: int = _SEED + 21903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rademacher_cplx",
            bench_rademacher_cplx(seed),
        )
    )


def bench_covering_number_family(
    seed: int = _SEED + 21904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "covering_number",
            bench_covering_number(seed),
        )
    )


def bench_metric_entropy_family(
    seed: int = _SEED + 21905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "metric_entropy",
            bench_metric_entropy(seed),
        )
    )
