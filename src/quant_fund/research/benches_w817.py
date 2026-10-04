"""Wave-817 Levy-fluctuation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.ladder_height import (
    bench_ladder_height,
)
from quant_fund.models.levy_fluct import (
    bench_levy_fluct,
)
from quant_fund.models.overshoot_levy import (
    bench_overshoot_levy,
)
from quant_fund.models.renewal_measure import (
    bench_renewal_measure,
)
from quant_fund.models.spitzer_levy import (
    bench_spitzer_levy,
)
from quant_fund.models.wiener_hopf_f import (
    bench_wiener_hopf_f,
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


def bench_wiener_hopf_f_family(
    seed: int = _SEED + 20500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wiener_hopf_f",
            bench_wiener_hopf_f(seed),
        )
    )


def bench_ladder_height_family(
    seed: int = _SEED + 20501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ladder_height",
            bench_ladder_height(seed),
        )
    )


def bench_renewal_measure_family(
    seed: int = _SEED + 20502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "renewal_measure",
            bench_renewal_measure(seed),
        )
    )


def bench_overshoot_levy_family(
    seed: int = _SEED + 20503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "overshoot_levy",
            bench_overshoot_levy(seed),
        )
    )


def bench_levy_fluct_family(
    seed: int = _SEED + 20504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "levy_fluct",
            bench_levy_fluct(seed),
        )
    )


def bench_spitzer_levy_family(
    seed: int = _SEED + 20505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spitzer_levy",
            bench_spitzer_levy(seed),
        )
    )
