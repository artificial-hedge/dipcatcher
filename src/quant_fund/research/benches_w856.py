"""Wave-856 quadrature/quasi-MC bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.halton_seq import (
    bench_halton_seq,
)
from quant_fund.models.latin_hypercube import (
    bench_latin_hypercube,
)
from quant_fund.models.monte_carlo_quad import (
    bench_monte_carlo_quad,
)
from quant_fund.models.quasi_mc import (
    bench_quasi_mc,
)
from quant_fund.models.sobol_seq import (
    bench_sobol_seq,
)
from quant_fund.models.stratified_mc import (
    bench_stratified_mc,
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


def bench_monte_carlo_quad_family(
    seed: int = _SEED + 24400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "monte_carlo_quad",
            bench_monte_carlo_quad(seed),
        )
    )


def bench_quasi_mc_family(
    seed: int = _SEED + 24401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quasi_mc",
            bench_quasi_mc(seed),
        )
    )


def bench_halton_seq_family(
    seed: int = _SEED + 24402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "halton_seq",
            bench_halton_seq(seed),
        )
    )


def bench_sobol_seq_family(
    seed: int = _SEED + 24403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sobol_seq",
            bench_sobol_seq(seed),
        )
    )


def bench_latin_hypercube_family(
    seed: int = _SEED + 24404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "latin_hypercube",
            bench_latin_hypercube(seed),
        )
    )


def bench_stratified_mc_family(
    seed: int = _SEED + 24405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stratified_mc",
            bench_stratified_mc(seed),
        )
    )
