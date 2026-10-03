"""Wave-806 jump-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.compound_poisson import (
    bench_compound_poisson,
)
from quant_fund.models.excursion_theory import (
    bench_excursion_theory,
)
from quant_fund.models.jump_diffusion import (
    bench_jump_diffusion,
)
from quant_fund.models.kou_model import (
    bench_kou_model,
)
from quant_fund.models.marked_hawkes import (
    bench_marked_hawkes,
)
from quant_fund.models.merton_jump import (
    bench_merton_jump,
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


def bench_jump_diffusion_family(
    seed: int = _SEED + 19500,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jump_diffusion",
            bench_jump_diffusion(seed),
        )
    )


def bench_merton_jump_family(
    seed: int = _SEED + 19501,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "merton_jump",
            bench_merton_jump(seed),
        )
    )


def bench_kou_model_family(
    seed: int = _SEED + 19502,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kou_model",
            bench_kou_model(seed),
        )
    )


def bench_compound_poisson_family(
    seed: int = _SEED + 19503,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "compound_poisson",
            bench_compound_poisson(seed),
        )
    )


def bench_excursion_theory_family(
    seed: int = _SEED + 19504,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "excursion_theory",
            bench_excursion_theory(seed),
        )
    )


def bench_marked_hawkes_family(
    seed: int = _SEED + 19505,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "marked_hawkes",
            bench_marked_hawkes(seed),
        )
    )
