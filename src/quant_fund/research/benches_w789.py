"""Wave-789 optimal-transport bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.benamou_brenier import (
    bench_benamou_brenier,
)
from quant_fund.models.entropy_regular import (
    bench_entropy_regular,
)
from quant_fund.models.fokker_planck2 import (
    bench_fokker_planck2,
)
from quant_fund.models.gradient_flow import (
    bench_gradient_flow,
)
from quant_fund.models.jko_step import bench_jko_step
from quant_fund.models.wasserstein_grad import (
    bench_wasserstein_grad,
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


def bench_wasserstein_grad_family(
    seed: int = _SEED + 17800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wasserstein_grad",
            bench_wasserstein_grad(seed),
        )
    )


def bench_jko_step_family(
    seed: int = _SEED + 17801,
) -> dict[str, float]:
    return _floats(_finite_blob("jko_step", bench_jko_step(seed)))


def bench_benamou_brenier_family(
    seed: int = _SEED + 17802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "benamou_brenier",
            bench_benamou_brenier(seed),
        )
    )


def bench_entropy_regular_family(
    seed: int = _SEED + 17803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "entropy_regular",
            bench_entropy_regular(seed),
        )
    )


def bench_fokker_planck2_family(
    seed: int = _SEED + 17804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fokker_planck2",
            bench_fokker_planck2(seed),
        )
    )


def bench_gradient_flow_family(
    seed: int = _SEED + 17805,
) -> dict[str, float]:
    return _floats(_finite_blob("gradient_flow", bench_gradient_flow(seed)))
