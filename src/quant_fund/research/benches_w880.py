"""Wave-880 quadrature/cubature bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adaptive_quad2 import (
    bench_adaptive_quad2,
)
from quant_fund.models.cubature_rule import (
    bench_cubature_rule,
)
from quant_fund.models.empirical_interp import (
    bench_empirical_interp,
)
from quant_fund.models.gq_adaptive import (
    bench_gq_adaptive,
)
from quant_fund.models.pod_deim import (
    bench_pod_deim,
)
from quant_fund.models.tensor_interp import (
    bench_tensor_interp,
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


def bench_gq_adaptive_family(
    seed: int = _SEED + 26800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gq_adaptive",
            bench_gq_adaptive(seed),
        )
    )


def bench_adaptive_quad2_family(
    seed: int = _SEED + 26801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "adaptive_quad2",
            bench_adaptive_quad2(seed),
        )
    )


def bench_pod_deim_family(
    seed: int = _SEED + 26802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pod_deim",
            bench_pod_deim(seed),
        )
    )


def bench_empirical_interp_family(
    seed: int = _SEED + 26803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "empirical_interp",
            bench_empirical_interp(seed),
        )
    )


def bench_cubature_rule_family(
    seed: int = _SEED + 26804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cubature_rule",
            bench_cubature_rule(seed),
        )
    )


def bench_tensor_interp_family(
    seed: int = _SEED + 26805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tensor_interp",
            bench_tensor_interp(seed),
        )
    )
