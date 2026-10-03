"""Wave-895 root-finding bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aitken_steffensen import (
    bench_aitken_steffensen,
)
from quant_fund.models.bulirsch_stoer import (
    bench_bulirsch_stoer,
)
from quant_fund.models.muller_root import (
    bench_muller_root,
)
from quant_fund.models.regula_falsi import (
    bench_regula_falsi,
)
from quant_fund.models.richardson_limit import (
    bench_richardson_limit,
)
from quant_fund.models.secant_root import (
    bench_secant_root,
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


def bench_secant_root_family(
    seed: int = _SEED + 28300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "secant_root",
            bench_secant_root(seed),
        )
    )


def bench_regula_falsi_family(
    seed: int = _SEED + 28301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "regula_falsi",
            bench_regula_falsi(seed),
        )
    )


def bench_muller_root_family(
    seed: int = _SEED + 28302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "muller_root",
            bench_muller_root(seed),
        )
    )


def bench_aitken_steffensen_family(
    seed: int = _SEED + 28303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "aitken_steffensen",
            bench_aitken_steffensen(seed),
        )
    )


def bench_richardson_limit_family(
    seed: int = _SEED + 28304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "richardson_limit",
            bench_richardson_limit(seed),
        )
    )


def bench_bulirsch_stoer_family(
    seed: int = _SEED + 28305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bulirsch_stoer",
            bench_bulirsch_stoer(seed),
        )
    )
