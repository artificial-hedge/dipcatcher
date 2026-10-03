"""Wave-882 preconditioner/domain-decomposition bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.balanced_dd import (
    bench_balanced_dd,
)
from quant_fund.models.diagonal_scale import (
    bench_diagonal_scale,
)
from quant_fund.models.nonoverlap_dd import (
    bench_nonoverlap_dd,
)
from quant_fund.models.overlap_dd import (
    bench_overlap_dd,
)
from quant_fund.models.restrictive_dd import (
    bench_restrictive_dd,
)
from quant_fund.models.spai_precond import (
    bench_spai_precond,
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


def bench_spai_precond_family(
    seed: int = _SEED + 27000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spai_precond",
            bench_spai_precond(seed),
        )
    )


def bench_diagonal_scale_family(
    seed: int = _SEED + 27001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "diagonal_scale",
            bench_diagonal_scale(seed),
        )
    )


def bench_nonoverlap_dd_family(
    seed: int = _SEED + 27002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nonoverlap_dd",
            bench_nonoverlap_dd(seed),
        )
    )


def bench_overlap_dd_family(
    seed: int = _SEED + 27003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "overlap_dd",
            bench_overlap_dd(seed),
        )
    )


def bench_restrictive_dd_family(
    seed: int = _SEED + 27004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "restrictive_dd",
            bench_restrictive_dd(seed),
        )
    )


def bench_balanced_dd_family(
    seed: int = _SEED + 27005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "balanced_dd",
            bench_balanced_dd(seed),
        )
    )
