"""Wave-832 weak-convergence-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.continuous_map import (
    bench_continuous_map,
)
from quant_fund.models.delta_method import (
    bench_delta_method,
)
from quant_fund.models.empirical_bridge import (
    bench_empirical_bridge,
)
from quant_fund.models.kmt_approx import (
    bench_kmt_approx,
)
from quant_fund.models.porte_manteau import (
    bench_porte_manteau,
)
from quant_fund.models.skorohod_embed import (
    bench_skorohod_embed,
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


def bench_porte_manteau_family(
    seed: int = _SEED + 22000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "porte_manteau",
            bench_porte_manteau(seed),
        )
    )


def bench_continuous_map_family(
    seed: int = _SEED + 22001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "continuous_map",
            bench_continuous_map(seed),
        )
    )


def bench_delta_method_family(
    seed: int = _SEED + 22002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "delta_method",
            bench_delta_method(seed),
        )
    )


def bench_skorohod_embed_family(
    seed: int = _SEED + 22003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "skorohod_embed",
            bench_skorohod_embed(seed),
        )
    )


def bench_kmt_approx_family(
    seed: int = _SEED + 22004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kmt_approx",
            bench_kmt_approx(seed),
        )
    )


def bench_empirical_bridge_family(
    seed: int = _SEED + 22005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "empirical_bridge",
            bench_empirical_bridge(seed),
        )
    )
