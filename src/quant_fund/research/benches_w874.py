"""Wave-874 reliability-analysis bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.first_order_rel import (
    bench_first_order_rel,
)
from quant_fund.models.line_sampling import (
    bench_line_sampling,
)
from quant_fund.models.metamodel_rel import (
    bench_metamodel_rel,
)
from quant_fund.models.sorm_method import (
    bench_sorm_method,
)
from quant_fund.models.subset_sim import (
    bench_subset_sim,
)
from quant_fund.models.uq_reliability import (
    bench_uq_reliability,
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


def bench_uq_reliability_family(
    seed: int = _SEED + 26200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "uq_reliability",
            bench_uq_reliability(seed),
        )
    )


def bench_first_order_rel_family(
    seed: int = _SEED + 26201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "first_order_rel",
            bench_first_order_rel(seed),
        )
    )


def bench_sorm_method_family(
    seed: int = _SEED + 26202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sorm_method",
            bench_sorm_method(seed),
        )
    )


def bench_subset_sim_family(
    seed: int = _SEED + 26203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "subset_sim",
            bench_subset_sim(seed),
        )
    )


def bench_line_sampling_family(
    seed: int = _SEED + 26204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "line_sampling",
            bench_line_sampling(seed),
        )
    )


def bench_metamodel_rel_family(
    seed: int = _SEED + 26205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "metamodel_rel",
            bench_metamodel_rel(seed),
        )
    )
