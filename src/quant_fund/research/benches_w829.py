"""Wave-829 continuous-martingale bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bounded_mart import (
    bench_bounded_mart,
)
from quant_fund.models.cadlag_mart import (
    bench_cadlag_mart,
)
from quant_fund.models.decomp_mart import (
    bench_decomp_mart,
)
from quant_fund.models.fv_mart import (
    bench_fv_mart,
)
from quant_fund.models.local_time_process import (
    bench_local_time_process,
)
from quant_fund.models.locator_proc import (
    bench_locator_proc,
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


def bench_local_time_process_family(
    seed: int = _SEED + 21700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "local_time_process",
            bench_local_time_process(seed),
        )
    )


def bench_bounded_mart_family(
    seed: int = _SEED + 21701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bounded_mart",
            bench_bounded_mart(seed),
        )
    )


def bench_fv_mart_family(
    seed: int = _SEED + 21702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fv_mart",
            bench_fv_mart(seed),
        )
    )


def bench_cadlag_mart_family(
    seed: int = _SEED + 21703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cadlag_mart",
            bench_cadlag_mart(seed),
        )
    )


def bench_locator_proc_family(
    seed: int = _SEED + 21704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "locator_proc",
            bench_locator_proc(seed),
        )
    )


def bench_decomp_mart_family(
    seed: int = _SEED + 21705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "decomp_mart",
            bench_decomp_mart(seed),
        )
    )
