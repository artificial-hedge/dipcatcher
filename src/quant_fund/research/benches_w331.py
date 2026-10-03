"""Wave-331 complexity-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.circuit_lb import bench_circuit_lb
from quant_fund.models.fpras_dnf import bench_fpras_dnf
from quant_fund.models.np_reduce import bench_np_reduce
from quant_fund.models.param_fpt import bench_param_fpt
from quant_fund.models.pcp_verify import bench_pcp_verify
from quant_fund.models.sumcheck import bench_sumcheck

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


def bench_np_reduce_family(seed: int = _SEED + 1893) -> dict[str, float]:
    return _floats(_finite_blob("np_reduce", bench_np_reduce(seed)))


def bench_fpras_dnf_family(seed: int = _SEED + 1894) -> dict[str, float]:
    return _floats(_finite_blob("fpras_dnf", bench_fpras_dnf(seed)))


def bench_sumcheck_family(seed: int = _SEED + 1895) -> dict[str, float]:
    return _floats(_finite_blob("sumcheck", bench_sumcheck(seed)))


def bench_param_fpt_family(seed: int = _SEED + 1896) -> dict[str, float]:
    return _floats(_finite_blob("param_fpt", bench_param_fpt(seed)))


def bench_pcp_verify_family(seed: int = _SEED + 1897) -> dict[str, float]:
    return _floats(_finite_blob("pcp_verify", bench_pcp_verify(seed)))


def bench_circuit_lb_family(seed: int = _SEED + 1898) -> dict[str, float]:
    return _floats(_finite_blob("circuit_lb", bench_circuit_lb(seed)))
