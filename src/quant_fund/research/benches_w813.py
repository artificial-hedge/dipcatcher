"""Wave-813 diffusion-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.diffusion_semigroup import (
    bench_diffusion_semigroup,
)
from quant_fund.models.feller_boundary import (
    bench_feller_boundary,
)
from quant_fund.models.kreyn_resolvent import (
    bench_kreyn_resolvent,
)
from quant_fund.models.scale_measure import (
    bench_scale_measure,
)
from quant_fund.models.speed_measure import (
    bench_speed_measure,
)
from quant_fund.models.yosida_op import (
    bench_yosida_op,
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


def bench_feller_boundary_family(
    seed: int = _SEED + 20100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "feller_boundary",
            bench_feller_boundary(seed),
        )
    )


def bench_scale_measure_family(
    seed: int = _SEED + 20101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "scale_measure",
            bench_scale_measure(seed),
        )
    )


def bench_speed_measure_family(
    seed: int = _SEED + 20102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "speed_measure",
            bench_speed_measure(seed),
        )
    )


def bench_diffusion_semigroup_family(
    seed: int = _SEED + 20103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "diffusion_semigroup",
            bench_diffusion_semigroup(seed),
        )
    )


def bench_yosida_op_family(
    seed: int = _SEED + 20104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "yosida_op",
            bench_yosida_op(seed),
        )
    )


def bench_kreyn_resolvent_family(
    seed: int = _SEED + 20105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kreyn_resolvent",
            bench_kreyn_resolvent(seed),
        )
    )
