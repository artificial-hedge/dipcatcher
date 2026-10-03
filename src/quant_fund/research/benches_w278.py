"""Wave-278 econ-models-2 benches: macro + dynamic equilibrium."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cobweb_model import bench_cobweb_model
from quant_fund.models.nk_phillips import bench_nk_phillips
from quant_fund.models.olg_model import bench_olg_model
from quant_fund.models.rbc_sim import bench_rbc_sim
from quant_fund.models.solow_model import bench_solow_model
from quant_fund.models.taylor_rule import bench_taylor_rule

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_rbc_sim_family(seed: int = _SEED + 1550) -> dict[str, float]:
    return _floats(_finite_blob("rbc_sim", bench_rbc_sim(seed)))


def bench_nk_phillips_family(seed: int = _SEED + 1551) -> dict[str, float]:
    return _floats(_finite_blob("nk_phillips", bench_nk_phillips(seed)))


def bench_taylor_rule_family(seed: int = _SEED + 1552) -> dict[str, float]:
    return _floats(_finite_blob("taylor_rule", bench_taylor_rule(seed)))


def bench_solow_model_family(seed: int = _SEED + 1553) -> dict[str, float]:
    return _floats(_finite_blob("solow_model", bench_solow_model(seed)))


def bench_olg_model_family(seed: int = _SEED + 1554) -> dict[str, float]:
    return _floats(_finite_blob("olg_model", bench_olg_model(seed)))


def bench_cobweb_model_family(seed: int = _SEED + 1555) -> dict[str, float]:
    return _floats(_finite_blob("cobweb_model", bench_cobweb_model(seed)))
