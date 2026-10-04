"""Wave-790 McKean-Vlasov bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.kac_theorem import (
    bench_kac_theorem,
)
from quant_fund.models.mckean_vlasov import (
    bench_mckean_vlasov,
)
from quant_fund.models.mean_field_game2 import (
    bench_mean_field_game2,
)
from quant_fund.models.nonlinear_markov import (
    bench_nonlinear_markov,
)
from quant_fund.models.propagation_chaos import (
    bench_propagation_chaos,
)
from quant_fund.models.self_stabilizing import (
    bench_self_stabilizing,
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


def bench_mckean_vlasov_family(
    seed: int = _SEED + 17900,
) -> dict[str, float]:
    return _floats(_finite_blob("mckean_vlasov", bench_mckean_vlasov(seed)))


def bench_mean_field_game2_family(
    seed: int = _SEED + 17901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mean_field_game2",
            bench_mean_field_game2(seed),
        )
    )


def bench_propagation_chaos_family(
    seed: int = _SEED + 17902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "propagation_chaos",
            bench_propagation_chaos(seed),
        )
    )


def bench_kac_theorem_family(
    seed: int = _SEED + 17903,
) -> dict[str, float]:
    return _floats(_finite_blob("kac_theorem", bench_kac_theorem(seed)))


def bench_nonlinear_markov_family(
    seed: int = _SEED + 17904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nonlinear_markov",
            bench_nonlinear_markov(seed),
        )
    )


def bench_self_stabilizing_family(
    seed: int = _SEED + 17905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "self_stabilizing",
            bench_self_stabilizing(seed),
        )
    )
