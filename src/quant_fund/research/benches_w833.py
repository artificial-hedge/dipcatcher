"""Wave-833 functional-limit-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brownian_approx import (
    bench_brownian_approx,
)
from quant_fund.models.donsker_invariance import (
    bench_donsker_invariance,
)
from quant_fund.models.fclt_invariance import (
    bench_fclt_invariance,
)
from quant_fund.models.martingale_fclt import (
    bench_martingale_fclt,
)
from quant_fund.models.stable_limit import (
    bench_stable_limit,
)
from quant_fund.models.strassen_flln import (
    bench_strassen_flln,
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


def bench_fclt_invariance_family(
    seed: int = _SEED + 22100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fclt_invariance",
            bench_fclt_invariance(seed),
        )
    )


def bench_donsker_invariance_family(
    seed: int = _SEED + 22101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "donsker_invariance",
            bench_donsker_invariance(seed),
        )
    )


def bench_martingale_fclt_family(
    seed: int = _SEED + 22102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "martingale_fclt",
            bench_martingale_fclt(seed),
        )
    )


def bench_stable_limit_family(
    seed: int = _SEED + 22103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_limit",
            bench_stable_limit(seed),
        )
    )


def bench_brownian_approx_family(
    seed: int = _SEED + 22104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "brownian_approx",
            bench_brownian_approx(seed),
        )
    )


def bench_strassen_flln_family(
    seed: int = _SEED + 22105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "strassen_flln",
            bench_strassen_flln(seed),
        )
    )
