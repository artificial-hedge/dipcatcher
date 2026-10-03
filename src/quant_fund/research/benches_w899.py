"""Wave-899 interpolation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bernstein_form import (
    bench_bernstein_form,
)
from quant_fund.models.cardinal_interp import (
    bench_cardinal_interp,
)
from quant_fund.models.chebyshev_interp import (
    bench_chebyshev_interp,
)
from quant_fund.models.osculating_interp import (
    bench_osculating_interp,
)
from quant_fund.models.rational_interp import (
    bench_rational_interp,
)
from quant_fund.models.shanks_trans import (
    bench_shanks_trans,
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


def bench_cardinal_interp_family(
    seed: int = _SEED + 28700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cardinal_interp",
            bench_cardinal_interp(seed),
        )
    )


def bench_bernstein_form_family(
    seed: int = _SEED + 28701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bernstein_form",
            bench_bernstein_form(seed),
        )
    )


def bench_shanks_trans_family(
    seed: int = _SEED + 28702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "shanks_trans",
            bench_shanks_trans(seed),
        )
    )


def bench_chebyshev_interp_family(
    seed: int = _SEED + 28703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chebyshev_interp",
            bench_chebyshev_interp(seed),
        )
    )


def bench_osculating_interp_family(
    seed: int = _SEED + 28704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "osculating_interp",
            bench_osculating_interp(seed),
        )
    )


def bench_rational_interp_family(
    seed: int = _SEED + 28705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rational_interp",
            bench_rational_interp(seed),
        )
    )
