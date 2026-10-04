"""Wave-830 concentration-of-measure bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.azuma_ineq import (
    bench_azuma_ineq,
)
from quant_fund.models.bounded_diff import (
    bench_bounded_diff,
)
from quant_fund.models.efron_stein import (
    bench_efron_stein,
)
from quant_fund.models.hoeffding_ineq import (
    bench_hoeffding_ineq,
)
from quant_fund.models.mcdiarmid_ineq import (
    bench_mcdiarmid_ineq,
)
from quant_fund.models.talagrand_ineq import (
    bench_talagrand_ineq,
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


def bench_azuma_ineq_family(
    seed: int = _SEED + 21800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "azuma_ineq",
            bench_azuma_ineq(seed),
        )
    )


def bench_mcdiarmid_ineq_family(
    seed: int = _SEED + 21801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mcdiarmid_ineq",
            bench_mcdiarmid_ineq(seed),
        )
    )


def bench_talagrand_ineq_family(
    seed: int = _SEED + 21802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "talagrand_ineq",
            bench_talagrand_ineq(seed),
        )
    )


def bench_efron_stein_family(
    seed: int = _SEED + 21803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "efron_stein",
            bench_efron_stein(seed),
        )
    )


def bench_bounded_diff_family(
    seed: int = _SEED + 21804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bounded_diff",
            bench_bounded_diff(seed),
        )
    )


def bench_hoeffding_ineq_family(
    seed: int = _SEED + 21805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hoeffding_ineq",
            bench_hoeffding_ineq(seed),
        )
    )
