"""Wave-582 geometric-Langlands-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.arinkin_gaitsgory import (
    bench_arinkin_gaitsgory,
)
from quant_fund.models.derived_satake import (
    bench_derived_satake,
)
from quant_fund.models.fusion_product import (
    bench_fusion_product,
)
from quant_fund.models.geometric_satake2 import (
    bench_geometric_satake2,
)
from quant_fund.models.nilp_cone import bench_nilp_cone
from quant_fund.models.spectral_bung import bench_spectral_bung

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


def bench_arinkin_gaitsgory_family(
    seed: int = _SEED + 3398,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "arinkin_gaitsgory",
            bench_arinkin_gaitsgory(seed),
        )
    )


def bench_derived_satake_family(
    seed: int = _SEED + 3399,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_satake",
            bench_derived_satake(seed),
        )
    )


def bench_spectral_bung_family(
    seed: int = _SEED + 3400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_bung",
            bench_spectral_bung(seed),
        )
    )


def bench_nilp_cone_family(seed: int = _SEED + 3401) -> dict[str, float]:
    return _floats(_finite_blob("nilp_cone", bench_nilp_cone(seed)))


def bench_geometric_satake2_family(
    seed: int = _SEED + 3402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "geometric_satake2",
            bench_geometric_satake2(seed),
        )
    )


def bench_fusion_product_family(
    seed: int = _SEED + 3403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fusion_product",
            bench_fusion_product(seed),
        )
    )
