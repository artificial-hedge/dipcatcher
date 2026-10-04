"""Wave-892 adaptive-mesh bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.form_analysis import (
    bench_form_analysis,
)
from quant_fund.models.greedy_marking import (
    bench_greedy_marking,
)
from quant_fund.models.hp_adaptive import (
    bench_hp_adaptive,
)
from quant_fund.models.residual_marking import (
    bench_residual_marking,
)
from quant_fund.models.space_time_adapt import (
    bench_space_time_adapt,
)
from quant_fund.models.wavelet_adapt import (
    bench_wavelet_adapt,
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


def bench_space_time_adapt_family(
    seed: int = _SEED + 28000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "space_time_adapt",
            bench_space_time_adapt(seed),
        )
    )


def bench_greedy_marking_family(
    seed: int = _SEED + 28001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "greedy_marking",
            bench_greedy_marking(seed),
        )
    )


def bench_form_analysis_family(
    seed: int = _SEED + 28002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "form_analysis",
            bench_form_analysis(seed),
        )
    )


def bench_hp_adaptive_family(
    seed: int = _SEED + 28003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hp_adaptive",
            bench_hp_adaptive(seed),
        )
    )


def bench_wavelet_adapt_family(
    seed: int = _SEED + 28004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wavelet_adapt",
            bench_wavelet_adapt(seed),
        )
    )


def bench_residual_marking_family(
    seed: int = _SEED + 28005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "residual_marking",
            bench_residual_marking(seed),
        )
    )
