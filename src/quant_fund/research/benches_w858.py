"""Wave-858 adaptive/oscillatory-quadrature bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adaptive_simpsons import (
    bench_adaptive_simpsons,
)
from quant_fund.models.double_exp_quad import (
    bench_double_exp_quad,
)
from quant_fund.models.filon_quad import (
    bench_filon_quad,
)
from quant_fund.models.levin_quad import (
    bench_levin_quad,
)
from quant_fund.models.osc_singular import (
    bench_osc_singular,
)
from quant_fund.models.tanh_sinh import (
    bench_tanh_sinh,
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


def bench_adaptive_simpsons_family(
    seed: int = _SEED + 24600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "adaptive_simpsons",
            bench_adaptive_simpsons(seed),
        )
    )


def bench_tanh_sinh_family(
    seed: int = _SEED + 24601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tanh_sinh",
            bench_tanh_sinh(seed),
        )
    )


def bench_double_exp_quad_family(
    seed: int = _SEED + 24602,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "double_exp_quad",
            bench_double_exp_quad(seed),
        )
    )


def bench_osc_singular_family(
    seed: int = _SEED + 24603,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "osc_singular",
            bench_osc_singular(seed),
        )
    )


def bench_filon_quad_family(
    seed: int = _SEED + 24604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "filon_quad",
            bench_filon_quad(seed),
        )
    )


def bench_levin_quad_family(
    seed: int = _SEED + 24605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "levin_quad",
            bench_levin_quad(seed),
        )
    )
