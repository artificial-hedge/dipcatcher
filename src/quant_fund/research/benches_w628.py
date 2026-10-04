"""Wave-628 operads-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dendroidal_seg import (
    bench_dendroidal_seg,
)
from quant_fund.models.higher_operad import (
    bench_higher_operad,
)
from quant_fund.models.moerdijk_weiss import (
    bench_moerdijk_weiss,
)
from quant_fund.models.operad_cat2 import (
    bench_operad_cat2,
)
from quant_fund.models.operad_infty3 import (
    bench_operad_infty3,
)
from quant_fund.models.operad_module import (
    bench_operad_module,
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


def bench_moerdijk_weiss_family(
    seed: int = _SEED + 3674,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "moerdijk_weiss",
            bench_moerdijk_weiss(seed),
        )
    )


def bench_higher_operad_family(
    seed: int = _SEED + 3675,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "higher_operad",
            bench_higher_operad(seed),
        )
    )


def bench_operad_infty3_family(
    seed: int = _SEED + 3676,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operad_infty3",
            bench_operad_infty3(seed),
        )
    )


def bench_operad_cat2_family(
    seed: int = _SEED + 3677,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operad_cat2",
            bench_operad_cat2(seed),
        )
    )


def bench_dendroidal_seg_family(
    seed: int = _SEED + 3678,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dendroidal_seg",
            bench_dendroidal_seg(seed),
        )
    )


def bench_operad_module_family(
    seed: int = _SEED + 3679,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operad_module",
            bench_operad_module(seed),
        )
    )
