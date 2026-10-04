"""Wave-805 stochastic-calculus bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.doss_sussmann import (
    bench_doss_sussmann,
)
from quant_fund.models.follmer_strat import (
    bench_follmer_strat,
)
from quant_fund.models.ito_isometry import (
    bench_ito_isometry,
)
from quant_fund.models.skorohod_lemma import (
    bench_skorohod_lemma,
)
from quant_fund.models.stratonovich_conv import (
    bench_stratonovich_conv,
)
from quant_fund.models.tanaka_meyer import (
    bench_tanaka_meyer,
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


def bench_ito_isometry_family(
    seed: int = _SEED + 19400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ito_isometry",
            bench_ito_isometry(seed),
        )
    )


def bench_stratonovich_conv_family(
    seed: int = _SEED + 19401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stratonovich_conv",
            bench_stratonovich_conv(seed),
        )
    )


def bench_tanaka_meyer_family(
    seed: int = _SEED + 19402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tanaka_meyer",
            bench_tanaka_meyer(seed),
        )
    )


def bench_follmer_strat_family(
    seed: int = _SEED + 19403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "follmer_strat",
            bench_follmer_strat(seed),
        )
    )


def bench_skorohod_lemma_family(
    seed: int = _SEED + 19404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "skorohod_lemma",
            bench_skorohod_lemma(seed),
        )
    )


def bench_doss_sussmann_family(
    seed: int = _SEED + 19405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "doss_sussmann",
            bench_doss_sussmann(seed),
        )
    )
