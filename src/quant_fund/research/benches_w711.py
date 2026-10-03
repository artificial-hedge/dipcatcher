"""Wave-711 homotopy-31 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.homotopy_local import (
    bench_homotopy_local,
)
from quant_fund.models.homotopy_sheaf2 import (
    bench_homotopy_sheaf2,
)
from quant_fund.models.homotopy_stable4 import (
    bench_homotopy_stable4,
)
from quant_fund.models.stable_coalgebra import (
    bench_stable_coalgebra,
)
from quant_fund.models.stable_inf_cat import (
    bench_stable_inf_cat,
)
from quant_fund.models.stable_sheaf2 import (
    bench_stable_sheaf2,
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


def bench_homotopy_sheaf2_family(
    seed: int = _SEED + 10000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_sheaf2",
            bench_homotopy_sheaf2(seed),
        )
    )


def bench_stable_inf_cat_family(
    seed: int = _SEED + 10001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_inf_cat",
            bench_stable_inf_cat(seed),
        )
    )


def bench_homotopy_stable4_family(
    seed: int = _SEED + 10002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_stable4",
            bench_homotopy_stable4(seed),
        )
    )


def bench_homotopy_local_family(
    seed: int = _SEED + 10003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "homotopy_local",
            bench_homotopy_local(seed),
        )
    )


def bench_stable_sheaf2_family(
    seed: int = _SEED + 10004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_sheaf2",
            bench_stable_sheaf2(seed),
        )
    )


def bench_stable_coalgebra_family(
    seed: int = _SEED + 10005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "stable_coalgebra",
            bench_stable_coalgebra(seed),
        )
    )
