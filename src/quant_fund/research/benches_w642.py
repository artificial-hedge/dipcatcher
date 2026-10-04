"""Wave-642 prismatic-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bhatt_scholze import (
    bench_bhatt_scholze,
)
from quant_fund.models.derived_prism import (
    bench_derived_prism,
)
from quant_fund.models.prismatic_dieudonne import (
    bench_prismatic_dieudonne,
)
from quant_fund.models.prismatic_f import (
    bench_prismatic_f,
)
from quant_fund.models.q_crystal import bench_q_crystal
from quant_fund.models.q_prism import bench_q_prism

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


def bench_prismatic_f_family(
    seed: int = _SEED + 3758,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "prismatic_f",
            bench_prismatic_f(seed),
        )
    )


def bench_bhatt_scholze_family(
    seed: int = _SEED + 3759,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "bhatt_scholze",
            bench_bhatt_scholze(seed),
        )
    )


def bench_q_crystal_family(
    seed: int = _SEED + 3760,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "q_crystal",
            bench_q_crystal(seed),
        )
    )


def bench_prismatic_dieudonne_family(
    seed: int = _SEED + 3761,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "prismatic_dieudonne",
            bench_prismatic_dieudonne(seed),
        )
    )


def bench_q_prism_family(
    seed: int = _SEED + 3762,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "q_prism",
            bench_q_prism(seed),
        )
    )


def bench_derived_prism_family(
    seed: int = _SEED + 3763,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "derived_prism",
            bench_derived_prism(seed),
        )
    )
