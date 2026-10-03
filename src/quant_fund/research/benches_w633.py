"""Wave-633 motivic-11 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cone_theorem import (
    bench_cone_theorem,
)
from quant_fund.models.contr_rational import (
    bench_contr_rational,
)
from quant_fund.models.motivic_abelian import (
    bench_motivic_abelian,
)
from quant_fund.models.motivic_coho2 import (
    bench_motivic_coho2,
)
from quant_fund.models.motivic_compact import (
    bench_motivic_compact,
)
from quant_fund.models.motivic_landweber import (
    bench_motivic_landweber,
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


def bench_motivic_coho2_family(
    seed: int = _SEED + 3704,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_coho2",
            bench_motivic_coho2(seed),
        )
    )


def bench_cone_theorem_family(
    seed: int = _SEED + 3705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cone_theorem",
            bench_cone_theorem(seed),
        )
    )


def bench_motivic_landweber_family(
    seed: int = _SEED + 3706,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_landweber",
            bench_motivic_landweber(seed),
        )
    )


def bench_motivic_abelian_family(
    seed: int = _SEED + 3707,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_abelian",
            bench_motivic_abelian(seed),
        )
    )


def bench_motivic_compact_family(
    seed: int = _SEED + 3708,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_compact",
            bench_motivic_compact(seed),
        )
    )


def bench_contr_rational_family(
    seed: int = _SEED + 3709,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "contr_rational",
            bench_contr_rational(seed),
        )
    )
