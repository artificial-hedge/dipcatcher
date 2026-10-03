"""Wave-807 optimal-stopping bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cayley_moser import (
    bench_cayley_moser,
)
from quant_fund.models.chow_robbins import (
    bench_chow_robbins,
)
from quant_fund.models.free_boundary import (
    bench_free_boundary,
)
from quant_fund.models.markov_stopping import (
    bench_markov_stopping,
)
from quant_fund.models.secretary_dp import (
    bench_secretary_dp,
)
from quant_fund.models.snell_envelope import (
    bench_snell_envelope,
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


def bench_snell_envelope_family(
    seed: int = _SEED + 19600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "snell_envelope",
            bench_snell_envelope(seed),
        )
    )


def bench_secretary_dp_family(
    seed: int = _SEED + 19601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "secretary_dp",
            bench_secretary_dp(seed),
        )
    )


def bench_cayley_moser_family(
    seed: int = _SEED + 19602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cayley_moser",
            bench_cayley_moser(seed),
        )
    )


def bench_chow_robbins_family(
    seed: int = _SEED + 19603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chow_robbins",
            bench_chow_robbins(seed),
        )
    )


def bench_markov_stopping_family(
    seed: int = _SEED + 19604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "markov_stopping",
            bench_markov_stopping(seed),
        )
    )


def bench_free_boundary_family(
    seed: int = _SEED + 19605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "free_boundary",
            bench_free_boundary(seed),
        )
    )
