"""Wave-800 stochastic-vol bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bates_model import (
    bench_bates_model,
)
from quant_fund.models.heston_model import (
    bench_heston_model,
)
from quant_fund.models.rough_heston import (
    bench_rough_heston,
)
from quant_fund.models.sabr_model import (
    bench_sabr_model,
)
from quant_fund.models.scott_vol import (
    bench_scott_vol,
)
from quant_fund.models.three_two_vol import (
    bench_three_two_vol,
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


def bench_heston_model_family(
    seed: int = _SEED + 18900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "heston_model",
            bench_heston_model(seed),
        )
    )


def bench_bates_model_family(
    seed: int = _SEED + 18901,
) -> dict[str, float]:
    return _floats(_finite_blob("bates_model", bench_bates_model(seed)))


def bench_rough_heston_family(
    seed: int = _SEED + 18902,
) -> dict[str, float]:
    return _floats(_finite_blob("rough_heston", bench_rough_heston(seed)))


def bench_sabr_model_family(
    seed: int = _SEED + 18903,
) -> dict[str, float]:
    return _floats(_finite_blob("sabr_model", bench_sabr_model(seed)))


def bench_three_two_vol_family(
    seed: int = _SEED + 18904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "three_two_vol",
            bench_three_two_vol(seed),
        )
    )


def bench_scott_vol_family(
    seed: int = _SEED + 18905,
) -> dict[str, float]:
    return _floats(_finite_blob("scott_vol", bench_scott_vol(seed)))
