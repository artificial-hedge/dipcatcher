"""Wave-859 meshfree/moving-least-squares bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.diffuse_element import (
    bench_diffuse_element,
)
from quant_fund.models.hp_clouds import (
    bench_hp_clouds,
)
from quant_fund.models.meshless_local import (
    bench_meshless_local,
)
from quant_fund.models.mls_shape import (
    bench_mls_shape,
)
from quant_fund.models.moving_least_sq import (
    bench_moving_least_sq,
)
from quant_fund.models.point_cloud_interp import (
    bench_point_cloud_interp,
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


def bench_moving_least_sq_family(
    seed: int = _SEED + 24700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "moving_least_sq",
            bench_moving_least_sq(seed),
        )
    )


def bench_mls_shape_family(
    seed: int = _SEED + 24701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mls_shape",
            bench_mls_shape(seed),
        )
    )


def bench_hp_clouds_family(
    seed: int = _SEED + 24702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hp_clouds",
            bench_hp_clouds(seed),
        )
    )


def bench_meshless_local_family(
    seed: int = _SEED + 24703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "meshless_local",
            bench_meshless_local(seed),
        )
    )


def bench_point_cloud_interp_family(
    seed: int = _SEED + 24704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "point_cloud_interp",
            bench_point_cloud_interp(seed),
        )
    )


def bench_diffuse_element_family(
    seed: int = _SEED + 24705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "diffuse_element",
            bench_diffuse_element(seed),
        )
    )
