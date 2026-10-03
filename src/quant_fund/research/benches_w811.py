"""Wave-811 point-process bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cambrian_pp import (
    bench_cambrian_pp,
)
from quant_fund.models.ergodic_pp import (
    bench_ergodic_pp,
)
from quant_fund.models.gneding_metric import (
    bench_gneding_metric,
)
from quant_fund.models.j_function import (
    bench_j_function,
)
from quant_fund.models.papangelou import (
    bench_papangelou,
)
from quant_fund.models.void_prob import (
    bench_void_prob,
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


def bench_cambrian_pp_family(
    seed: int = _SEED + 19900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cambrian_pp",
            bench_cambrian_pp(seed),
        )
    )


def bench_papangelou_family(
    seed: int = _SEED + 19901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "papangelou",
            bench_papangelou(seed),
        )
    )


def bench_gneding_metric_family(
    seed: int = _SEED + 19902,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gneding_metric",
            bench_gneding_metric(seed),
        )
    )


def bench_void_prob_family(
    seed: int = _SEED + 19903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "void_prob",
            bench_void_prob(seed),
        )
    )


def bench_j_function_family(
    seed: int = _SEED + 19904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "j_function",
            bench_j_function(seed),
        )
    )


def bench_ergodic_pp_family(
    seed: int = _SEED + 19905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ergodic_pp",
            bench_ergodic_pp(seed),
        )
    )
