"""Wave-846 special-functions bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.airy_fn import (
    bench_airy_fn,
)
from quant_fund.models.bessel_fn import (
    bench_bessel_fn,
)
from quant_fund.models.beta_fn import (
    bench_beta_fn,
)
from quant_fund.models.error_fn import (
    bench_error_fn,
)
from quant_fund.models.gamma_fn import (
    bench_gamma_fn,
)
from quant_fund.models.hypergeometric_fn import (
    bench_hypergeometric_fn,
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


def bench_gamma_fn_family(
    seed: int = _SEED + 23400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gamma_fn",
            bench_gamma_fn(seed),
        )
    )


def bench_beta_fn_family(
    seed: int = _SEED + 23401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "beta_fn",
            bench_beta_fn(seed),
        )
    )


def bench_bessel_fn_family(
    seed: int = _SEED + 23402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bessel_fn",
            bench_bessel_fn(seed),
        )
    )


def bench_airy_fn_family(
    seed: int = _SEED + 23403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "airy_fn",
            bench_airy_fn(seed),
        )
    )


def bench_error_fn_family(
    seed: int = _SEED + 23404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "error_fn",
            bench_error_fn(seed),
        )
    )


def bench_hypergeometric_fn_family(
    seed: int = _SEED + 23405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hypergeometric_fn",
            bench_hypergeometric_fn(seed),
        )
    )
