"""Wave-818 stochastic-integration bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bounded_var import (
    bench_bounded_var,
)
from quant_fund.models.covariation import (
    bench_covariation,
)
from quant_fund.models.ito_integral import (
    bench_ito_integral,
)
from quant_fund.models.mart_meas import (
    bench_mart_meas,
)
from quant_fund.models.stochastic_int2 import (
    bench_stochastic_int2,
)
from quant_fund.models.vector_mart import (
    bench_vector_mart,
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


def bench_ito_integral_family(
    seed: int = _SEED + 20600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ito_integral",
            bench_ito_integral(seed),
        )
    )


def bench_mart_meas_family(
    seed: int = _SEED + 20601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mart_meas",
            bench_mart_meas(seed),
        )
    )


def bench_vector_mart_family(
    seed: int = _SEED + 20602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "vector_mart",
            bench_vector_mart(seed),
        )
    )


def bench_bounded_var_family(
    seed: int = _SEED + 20603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bounded_var",
            bench_bounded_var(seed),
        )
    )


def bench_stochastic_int2_family(
    seed: int = _SEED + 20604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_int2",
            bench_stochastic_int2(seed),
        )
    )


def bench_covariation_family(
    seed: int = _SEED + 20605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "covariation",
            bench_covariation(seed),
        )
    )
