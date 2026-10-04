"""Wave-840 rational-approximation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.loewner_interp import (
    bench_loewner_interp,
)
from quant_fund.models.nevanlinna_pick import (
    bench_nevanlinna_pick,
)
from quant_fund.models.pade_approx import (
    bench_pade_approx,
)
from quant_fund.models.rational_chebyshev import (
    bench_rational_chebyshev,
)
from quant_fund.models.schur_continued import (
    bench_schur_continued,
)
from quant_fund.models.stieltjes_fraction import (
    bench_stieltjes_fraction,
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


def bench_pade_approx_family(
    seed: int = _SEED + 22800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pade_approx",
            bench_pade_approx(seed),
        )
    )


def bench_rational_chebyshev_family(
    seed: int = _SEED + 22801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rational_chebyshev",
            bench_rational_chebyshev(seed),
        )
    )


def bench_stieltjes_fraction_family(
    seed: int = _SEED + 22802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stieltjes_fraction",
            bench_stieltjes_fraction(seed),
        )
    )


def bench_loewner_interp_family(
    seed: int = _SEED + 22803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "loewner_interp",
            bench_loewner_interp(seed),
        )
    )


def bench_nevanlinna_pick_family(
    seed: int = _SEED + 22804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nevanlinna_pick",
            bench_nevanlinna_pick(seed),
        )
    )


def bench_schur_continued_family(
    seed: int = _SEED + 22805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "schur_continued",
            bench_schur_continued(seed),
        )
    )
