"""Wave-809 Markov-chain theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cutoff_phenomenon import (
    bench_cutoff_phenomenon,
)
from quant_fund.models.doeblin_coupling import (
    bench_doeblin_coupling,
)
from quant_fund.models.drift_lyapunov import (
    bench_drift_lyapunov,
)
from quant_fund.models.ergodic_markov import (
    bench_ergodic_markov,
)
from quant_fund.models.harris_recurrent import (
    bench_harris_recurrent,
)
from quant_fund.models.mixing_time import (
    bench_mixing_time,
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


def bench_doeblin_coupling_family(
    seed: int = _SEED + 19800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "doeblin_coupling",
            bench_doeblin_coupling(seed),
        )
    )


def bench_harris_recurrent_family(
    seed: int = _SEED + 19801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "harris_recurrent",
            bench_harris_recurrent(seed),
        )
    )


def bench_ergodic_markov_family(
    seed: int = _SEED + 19802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ergodic_markov",
            bench_ergodic_markov(seed),
        )
    )


def bench_mixing_time_family(
    seed: int = _SEED + 19803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mixing_time",
            bench_mixing_time(seed),
        )
    )


def bench_drift_lyapunov_family(
    seed: int = _SEED + 19804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "drift_lyapunov",
            bench_drift_lyapunov(seed),
        )
    )


def bench_cutoff_phenomenon_family(
    seed: int = _SEED + 19805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cutoff_phenomenon",
            bench_cutoff_phenomenon(seed),
        )
    )
