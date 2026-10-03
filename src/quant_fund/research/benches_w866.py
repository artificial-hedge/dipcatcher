"""Wave-866 model-order-reduction bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.deim_point import (
    bench_deim_point,
)
from quant_fund.models.eim_interp import (
    bench_eim_interp,
)
from quant_fund.models.greedy_rb import (
    bench_greedy_rb,
)
from quant_fund.models.pod_galerkin import (
    bench_pod_galerkin,
)
from quant_fund.models.proper_gen import (
    bench_proper_gen,
)
from quant_fund.models.reduced_basis import (
    bench_reduced_basis,
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


def bench_pod_galerkin_family(
    seed: int = _SEED + 25400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pod_galerkin",
            bench_pod_galerkin(seed),
        )
    )


def bench_reduced_basis_family(
    seed: int = _SEED + 25401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "reduced_basis",
            bench_reduced_basis(seed),
        )
    )


def bench_deim_point_family(
    seed: int = _SEED + 25402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "deim_point",
            bench_deim_point(seed),
        )
    )


def bench_greedy_rb_family(
    seed: int = _SEED + 25403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "greedy_rb",
            bench_greedy_rb(seed),
        )
    )


def bench_eim_interp_family(
    seed: int = _SEED + 25404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "eim_interp",
            bench_eim_interp(seed),
        )
    )


def bench_proper_gen_family(
    seed: int = _SEED + 25405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "proper_gen",
            bench_proper_gen(seed),
        )
    )
