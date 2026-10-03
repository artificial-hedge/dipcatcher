"""Wave-814 stochastic-flow bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.karal_flow import (
    bench_karal_flow,
)
from quant_fund.models.kunita_flow import (
    bench_kunita_flow,
)
from quant_fund.models.liouville_flow import (
    bench_liouville_flow,
)
from quant_fund.models.meyers_process import (
    bench_meyers_process,
)
from quant_fund.models.stochastic_damping import (
    bench_stochastic_damping,
)
from quant_fund.models.stochastic_flow import (
    bench_stochastic_flow,
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


def bench_stochastic_flow_family(
    seed: int = _SEED + 20200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_flow",
            bench_stochastic_flow(seed),
        )
    )


def bench_kunita_flow_family(
    seed: int = _SEED + 20201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kunita_flow",
            bench_kunita_flow(seed),
        )
    )


def bench_liouville_flow_family(
    seed: int = _SEED + 20202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "liouville_flow",
            bench_liouville_flow(seed),
        )
    )


def bench_stochastic_damping_family(
    seed: int = _SEED + 20203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stochastic_damping",
            bench_stochastic_damping(seed),
        )
    )


def bench_meyers_process_family(
    seed: int = _SEED + 20204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "meyers_process",
            bench_meyers_process(seed),
        )
    )


def bench_karal_flow_family(
    seed: int = _SEED + 20205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "karal_flow",
            bench_karal_flow(seed),
        )
    )
