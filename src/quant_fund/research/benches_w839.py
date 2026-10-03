"""Wave-839 approximation-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bernstein_poly import (
    bench_bernstein_poly,
)
from quant_fund.models.chebyshev_alternation import (
    bench_chebyshev_alternation,
)
from quant_fund.models.fourier_decay import (
    bench_fourier_decay,
)
from quant_fund.models.jackson_direct import (
    bench_jackson_direct,
)
from quant_fund.models.kolmogorov_nwidth import (
    bench_kolmogorov_nwidth,
)
from quant_fund.models.markov_brothers import (
    bench_markov_brothers,
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


def bench_jackson_direct_family(
    seed: int = _SEED + 22700,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jackson_direct",
            bench_jackson_direct(seed),
        )
    )


def bench_chebyshev_alternation_family(
    seed: int = _SEED + 22701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "chebyshev_alternation",
            bench_chebyshev_alternation(seed),
        )
    )


def bench_kolmogorov_nwidth_family(
    seed: int = _SEED + 22702,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kolmogorov_nwidth",
            bench_kolmogorov_nwidth(seed),
        )
    )


def bench_bernstein_poly_family(
    seed: int = _SEED + 22703,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bernstein_poly",
            bench_bernstein_poly(seed),
        )
    )


def bench_markov_brothers_family(
    seed: int = _SEED + 22704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "markov_brothers",
            bench_markov_brothers(seed),
        )
    )


def bench_fourier_decay_family(
    seed: int = _SEED + 22705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fourier_decay",
            bench_fourier_decay(seed),
        )
    )
