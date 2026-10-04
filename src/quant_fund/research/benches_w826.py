"""Wave-826 maximal-inequality bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bj_ineq import (
    bench_bj_ineq,
)
from quant_fund.models.doob_ineq import (
    bench_doob_ineq,
)
from quant_fund.models.etemadi_ineq import (
    bench_etemadi_ineq,
)
from quant_fund.models.kolmogorov_ineq import (
    bench_kolmogorov_ineq,
)
from quant_fund.models.levy_ineq import (
    bench_levy_ineq,
)
from quant_fund.models.max_ineq import (
    bench_max_ineq,
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


def bench_doob_ineq_family(
    seed: int = _SEED + 21400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "doob_ineq",
            bench_doob_ineq(seed),
        )
    )


def bench_max_ineq_family(
    seed: int = _SEED + 21401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "max_ineq",
            bench_max_ineq(seed),
        )
    )


def bench_bj_ineq_family(
    seed: int = _SEED + 21402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bj_ineq",
            bench_bj_ineq(seed),
        )
    )


def bench_kolmogorov_ineq_family(
    seed: int = _SEED + 21403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kolmogorov_ineq",
            bench_kolmogorov_ineq(seed),
        )
    )


def bench_etemadi_ineq_family(
    seed: int = _SEED + 21404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etemadi_ineq",
            bench_etemadi_ineq(seed),
        )
    )


def bench_levy_ineq_family(
    seed: int = _SEED + 21405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "levy_ineq",
            bench_levy_ineq(seed),
        )
    )
