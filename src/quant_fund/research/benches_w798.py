"""Wave-798 forward-SDE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anticipating_sde import (
    bench_anticipating_sde,
)
from quant_fund.models.delayed_sde import (
    bench_delayed_sde,
)
from quant_fund.models.forward_sde import (
    bench_forward_sde,
)
from quant_fund.models.functional_sde import (
    bench_functional_sde,
)
from quant_fund.models.neutral_sde import (
    bench_neutral_sde,
)
from quant_fund.models.random_sde import (
    bench_random_sde,
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


def bench_forward_sde_family(
    seed: int = _SEED + 18700,
) -> dict[str, float]:
    return _floats(_finite_blob("forward_sde", bench_forward_sde(seed)))


def bench_random_sde_family(
    seed: int = _SEED + 18701,
) -> dict[str, float]:
    return _floats(_finite_blob("random_sde", bench_random_sde(seed)))


def bench_anticipating_sde_family(
    seed: int = _SEED + 18702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "anticipating_sde",
            bench_anticipating_sde(seed),
        )
    )


def bench_functional_sde_family(
    seed: int = _SEED + 18703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "functional_sde",
            bench_functional_sde(seed),
        )
    )


def bench_delayed_sde_family(
    seed: int = _SEED + 18704,
) -> dict[str, float]:
    return _floats(_finite_blob("delayed_sde", bench_delayed_sde(seed)))


def bench_neutral_sde_family(
    seed: int = _SEED + 18705,
) -> dict[str, float]:
    return _floats(_finite_blob("neutral_sde", bench_neutral_sde(seed)))
