"""Wave-867 inverse-problem bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bayes_inverse import (
    bench_bayes_inverse,
)
from quant_fund.models.iter_regularize import (
    bench_iter_regularize,
)
from quant_fund.models.l_curve_opt import (
    bench_l_curve_opt,
)
from quant_fund.models.morozov_dp import (
    bench_morozov_dp,
)
from quant_fund.models.tikhonov_reg import (
    bench_tikhonov_reg,
)
from quant_fund.models.tv_denoise import (
    bench_tv_denoise,
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


def bench_tikhonov_reg_family(
    seed: int = _SEED + 25500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tikhonov_reg",
            bench_tikhonov_reg(seed),
        )
    )


def bench_morozov_dp_family(
    seed: int = _SEED + 25501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "morozov_dp",
            bench_morozov_dp(seed),
        )
    )


def bench_l_curve_opt_family(
    seed: int = _SEED + 25502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "l_curve_opt",
            bench_l_curve_opt(seed),
        )
    )


def bench_iter_regularize_family(
    seed: int = _SEED + 25503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "iter_regularize",
            bench_iter_regularize(seed),
        )
    )


def bench_tv_denoise_family(
    seed: int = _SEED + 25504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tv_denoise",
            bench_tv_denoise(seed),
        )
    )


def bench_bayes_inverse_family(
    seed: int = _SEED + 25505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bayes_inverse",
            bench_bayes_inverse(seed),
        )
    )
