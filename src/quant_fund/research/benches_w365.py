"""Wave-365 probability-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.azuma import bench_azuma
from quant_fund.models.coupling_arg import bench_coupling_arg
from quant_fund.models.doob_decomp import bench_doob_decomp
from quant_fund.models.ergodic_thm import bench_ergodic_thm
from quant_fund.models.martingale_clt import bench_martingale_clt
from quant_fund.models.optional_stopping import bench_optional_stopping

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


def bench_optional_stopping_family(seed: int = _SEED + 2097) -> dict[str, float]:
    return _floats(_finite_blob("optional_stopping", bench_optional_stopping(seed)))


def bench_doob_decomp_family(seed: int = _SEED + 2098) -> dict[str, float]:
    return _floats(_finite_blob("doob_decomp", bench_doob_decomp(seed)))


def bench_martingale_clt_family(seed: int = _SEED + 2099) -> dict[str, float]:
    return _floats(_finite_blob("martingale_clt", bench_martingale_clt(seed)))


def bench_azuma_family(seed: int = _SEED + 2100) -> dict[str, float]:
    return _floats(_finite_blob("azuma", bench_azuma(seed)))


def bench_coupling_arg_family(seed: int = _SEED + 2101) -> dict[str, float]:
    return _floats(_finite_blob("coupling_arg", bench_coupling_arg(seed)))


def bench_ergodic_thm_family(seed: int = _SEED + 2102) -> dict[str, float]:
    return _floats(_finite_blob("ergodic_thm", bench_ergodic_thm(seed)))
