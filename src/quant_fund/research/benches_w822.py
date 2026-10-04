"""Wave-822 filtration bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.enlargement_f import (
    bench_enlargement_f,
)
from quant_fund.models.initial_enlarg import (
    bench_initial_enlarg,
)
from quant_fund.models.natural_filtration import (
    bench_natural_filtration,
)
from quant_fund.models.progressive_enlarg import (
    bench_progressive_enlarg,
)
from quant_fund.models.right_continuous_f import (
    bench_right_continuous_f,
)
from quant_fund.models.usual_aug import (
    bench_usual_aug,
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


def bench_natural_filtration_family(
    seed: int = _SEED + 21000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "natural_filtration",
            bench_natural_filtration(seed),
        )
    )


def bench_right_continuous_f_family(
    seed: int = _SEED + 21001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "right_continuous_f",
            bench_right_continuous_f(seed),
        )
    )


def bench_usual_aug_family(
    seed: int = _SEED + 21002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "usual_aug",
            bench_usual_aug(seed),
        )
    )


def bench_enlargement_f_family(
    seed: int = _SEED + 21003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "enlargement_f",
            bench_enlargement_f(seed),
        )
    )


def bench_initial_enlarg_family(
    seed: int = _SEED + 21004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "initial_enlarg",
            bench_initial_enlarg(seed),
        )
    )


def bench_progressive_enlarg_family(
    seed: int = _SEED + 21005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "progressive_enlarg",
            bench_progressive_enlarg(seed),
        )
    )
