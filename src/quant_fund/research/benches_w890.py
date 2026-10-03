"""Wave-890 special-function bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.clenshaw_quad import (
    bench_clenshaw_quad,
)
from quant_fund.models.elliptic_fn import (
    bench_elliptic_fn,
)
from quant_fund.models.fejer_nested import (
    bench_fejer_nested,
)
from quant_fund.models.hartley_transform import (
    bench_hartley_transform,
)
from quant_fund.models.radon_transform import (
    bench_radon_transform,
)
from quant_fund.models.zeta_fn import (
    bench_zeta_fn,
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


def bench_zeta_fn_family(
    seed: int = _SEED + 27800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zeta_fn",
            bench_zeta_fn(seed),
        )
    )


def bench_elliptic_fn_family(
    seed: int = _SEED + 27801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "elliptic_fn",
            bench_elliptic_fn(seed),
        )
    )


def bench_hartley_transform_family(
    seed: int = _SEED + 27802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hartley_transform",
            bench_hartley_transform(seed),
        )
    )


def bench_radon_transform_family(
    seed: int = _SEED + 27803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "radon_transform",
            bench_radon_transform(seed),
        )
    )


def bench_clenshaw_quad_family(
    seed: int = _SEED + 27804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "clenshaw_quad",
            bench_clenshaw_quad(seed),
        )
    )


def bench_fejer_nested_family(
    seed: int = _SEED + 27805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fejer_nested",
            bench_fejer_nested(seed),
        )
    )
