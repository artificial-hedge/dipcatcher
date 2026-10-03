"""Wave-639 homotopy-16 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.finite_chromatic import (
    bench_finite_chromatic,
)
from quant_fund.models.finite_htpy import (
    bench_finite_htpy,
)
from quant_fund.models.homotopy_fiber2 import (
    bench_homotopy_fiber2,
)
from quant_fund.models.periodic_htpy import (
    bench_periodic_htpy,
)
from quant_fund.models.rational_spec import (
    bench_rational_spec,
)
from quant_fund.models.stable_htpy2 import (
    bench_stable_htpy2,
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


def bench_homotopy_fiber2_family(
    seed: int = _SEED + 3740,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_fiber2",
            bench_homotopy_fiber2(seed),
        )
    )


def bench_stable_htpy2_family(
    seed: int = _SEED + 3741,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_htpy2",
            bench_stable_htpy2(seed),
        )
    )


def bench_finite_htpy_family(
    seed: int = _SEED + 3742,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "finite_htpy",
            bench_finite_htpy(seed),
        )
    )


def bench_rational_spec_family(
    seed: int = _SEED + 3743,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rational_spec",
            bench_rational_spec(seed),
        )
    )


def bench_finite_chromatic_family(
    seed: int = _SEED + 3744,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "finite_chromatic",
            bench_finite_chromatic(seed),
        )
    )


def bench_periodic_htpy_family(
    seed: int = _SEED + 3745,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "periodic_htpy",
            bench_periodic_htpy(seed),
        )
    )
