"""Wave-836 integral-geometry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.buffon_needle import (
    bench_buffon_needle,
)
from quant_fund.models.crofton_formula import (
    bench_crofton_formula,
)
from quant_fund.models.hadwiger_chars import (
    bench_hadwiger_chars,
)
from quant_fund.models.kinematic_measure import (
    bench_kinematic_measure,
)
from quant_fund.models.kubota_mean_width import (
    bench_kubota_mean_width,
)
from quant_fund.models.santalo_measure import (
    bench_santalo_measure,
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


def bench_crofton_formula_family(
    seed: int = _SEED + 22400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "crofton_formula",
            bench_crofton_formula(seed),
        )
    )


def bench_kinematic_measure_family(
    seed: int = _SEED + 22401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kinematic_measure",
            bench_kinematic_measure(seed),
        )
    )


def bench_buffon_needle_family(
    seed: int = _SEED + 22402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "buffon_needle",
            bench_buffon_needle(seed),
        )
    )


def bench_santalo_measure_family(
    seed: int = _SEED + 22403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "santalo_measure",
            bench_santalo_measure(seed),
        )
    )


def bench_kubota_mean_width_family(
    seed: int = _SEED + 22404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kubota_mean_width",
            bench_kubota_mean_width(seed),
        )
    )


def bench_hadwiger_chars_family(
    seed: int = _SEED + 22405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hadwiger_chars",
            bench_hadwiger_chars(seed),
        )
    )
