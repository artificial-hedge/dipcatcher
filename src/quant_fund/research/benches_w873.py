"""Wave-873 domain-decomposition bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.asm_precond import (
    bench_asm_precond,
)
from quant_fund.models.baldding_dd import (
    bench_baldding_dd,
)
from quant_fund.models.dd_partition import (
    bench_dd_partition,
)
from quant_fund.models.feti_dp import (
    bench_feti_dp,
)
from quant_fund.models.neumann_dd import (
    bench_neumann_dd,
)
from quant_fund.models.subspace_dd import (
    bench_subspace_dd,
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


def bench_dd_partition_family(
    seed: int = _SEED + 26100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dd_partition",
            bench_dd_partition(seed),
        )
    )


def bench_baldding_dd_family(
    seed: int = _SEED + 26101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "baldding_dd",
            bench_baldding_dd(seed),
        )
    )


def bench_neumann_dd_family(
    seed: int = _SEED + 26102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "neumann_dd",
            bench_neumann_dd(seed),
        )
    )


def bench_feti_dp_family(
    seed: int = _SEED + 26103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "feti_dp",
            bench_feti_dp(seed),
        )
    )


def bench_subspace_dd_family(
    seed: int = _SEED + 26104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "subspace_dd",
            bench_subspace_dd(seed),
        )
    )


def bench_asm_precond_family(
    seed: int = _SEED + 26105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "asm_precond",
            bench_asm_precond(seed),
        )
    )
