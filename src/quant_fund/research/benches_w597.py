"""Wave-597 chromatic-homotopy bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bousfield_kan import (
    bench_bousfield_kan,
)
from quant_fund.models.curtis_lower import bench_curtis_lower
from quant_fund.models.dror_smith import bench_dror_smith
from quant_fund.models.lannes_t import bench_lannes_t
from quant_fund.models.periodicity_thm import (
    bench_periodicity_thm,
)
from quant_fund.models.telescope_conj import (
    bench_telescope_conj,
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


def bench_curtis_lower_family(
    seed: int = _SEED + 3488,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "curtis_lower",
            bench_curtis_lower(seed),
        )
    )


def bench_bousfield_kan_family(
    seed: int = _SEED + 3489,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bousfield_kan",
            bench_bousfield_kan(seed),
        )
    )


def bench_lannes_t_family(seed: int = _SEED + 3490) -> dict[str, float]:
    return _floats(_finite_blob("lannes_t", bench_lannes_t(seed)))


def bench_dror_smith_family(
    seed: int = _SEED + 3491,
) -> dict[str, float]:
    return _floats(_finite_blob("dror_smith", bench_dror_smith(seed)))


def bench_telescope_conj_family(
    seed: int = _SEED + 3492,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "telescope_conj",
            bench_telescope_conj(seed),
        )
    )


def bench_periodicity_thm_family(
    seed: int = _SEED + 3493,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "periodicity_thm",
            bench_periodicity_thm(seed),
        )
    )
