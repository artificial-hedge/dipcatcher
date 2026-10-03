"""Wave-869 MC-variance-reduction bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.antithetic_var import (
    bench_antithetic_var,
)
from quant_fund.models.common_random import (
    bench_common_random,
)
from quant_fund.models.conditional_mc import (
    bench_conditional_mc,
)
from quant_fund.models.control_variate import (
    bench_control_variate,
)
from quant_fund.models.importance_sampling import (
    bench_importance_sampling,
)
from quant_fund.models.stratified_var import (
    bench_stratified_var,
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


def bench_antithetic_var_family(
    seed: int = _SEED + 25700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "antithetic_var",
            bench_antithetic_var(seed),
        )
    )


def bench_control_variate_family(
    seed: int = _SEED + 25701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "control_variate",
            bench_control_variate(seed),
        )
    )


def bench_importance_sampling_family(
    seed: int = _SEED + 25702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "importance_sampling",
            bench_importance_sampling(seed),
        )
    )


def bench_stratified_var_family(
    seed: int = _SEED + 25703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stratified_var",
            bench_stratified_var(seed),
        )
    )


def bench_common_random_family(
    seed: int = _SEED + 25704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "common_random",
            bench_common_random(seed),
        )
    )


def bench_conditional_mc_family(
    seed: int = _SEED + 25705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "conditional_mc",
            bench_conditional_mc(seed),
        )
    )
