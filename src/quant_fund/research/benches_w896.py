"""Wave-896 RK/IVP bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.backward_euler import (
    bench_backward_euler,
)
from quant_fund.models.bogacki_shampine import (
    bench_bogacki_shampine,
)
from quant_fund.models.cash_karp import (
    bench_cash_karp,
)
from quant_fund.models.dormand_prince import (
    bench_dormand_prince,
)
from quant_fund.models.fehlberg_rk import (
    bench_fehlberg_rk,
)
from quant_fund.models.predictor_corrector import (
    bench_predictor_corrector,
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


def bench_fehlberg_rk_family(
    seed: int = _SEED + 28400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fehlberg_rk",
            bench_fehlberg_rk(seed),
        )
    )


def bench_dormand_prince_family(
    seed: int = _SEED + 28401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dormand_prince",
            bench_dormand_prince(seed),
        )
    )


def bench_cash_karp_family(
    seed: int = _SEED + 28402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cash_karp",
            bench_cash_karp(seed),
        )
    )


def bench_bogacki_shampine_family(
    seed: int = _SEED + 28403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bogacki_shampine",
            bench_bogacki_shampine(seed),
        )
    )


def bench_backward_euler_family(
    seed: int = _SEED + 28404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "backward_euler",
            bench_backward_euler(seed),
        )
    )


def bench_predictor_corrector_family(
    seed: int = _SEED + 28405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "predictor_corrector",
            bench_predictor_corrector(seed),
        )
    )
