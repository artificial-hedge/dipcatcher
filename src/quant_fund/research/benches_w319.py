"""Wave-319 proof-automation canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.congruence_closure import bench_congruence_closure
from quant_fund.models.nelson_oppen import bench_nelson_oppen
from quant_fund.models.omega_lia import bench_omega_lia
from quant_fund.models.ring_normalize import bench_ring_normalize
from quant_fund.models.term_rewrite import bench_term_rewrite
from quant_fund.models.tseitin_cnf import bench_tseitin_cnf

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


def bench_congruence_closure_family(seed: int = _SEED + 1821) -> dict[str, float]:
    return _floats(_finite_blob("congruence_closure", bench_congruence_closure(seed)))


def bench_ring_normalize_family(seed: int = _SEED + 1822) -> dict[str, float]:
    return _floats(_finite_blob("ring_normalize", bench_ring_normalize(seed)))


def bench_omega_lia_family(seed: int = _SEED + 1823) -> dict[str, float]:
    return _floats(_finite_blob("omega_lia", bench_omega_lia(seed)))


def bench_nelson_oppen_family(seed: int = _SEED + 1824) -> dict[str, float]:
    return _floats(_finite_blob("nelson_oppen", bench_nelson_oppen(seed)))


def bench_term_rewrite_family(seed: int = _SEED + 1825) -> dict[str, float]:
    return _floats(_finite_blob("term_rewrite", bench_term_rewrite(seed)))


def bench_tseitin_cnf_family(seed: int = _SEED + 1826) -> dict[str, float]:
    return _floats(_finite_blob("tseitin_cnf", bench_tseitin_cnf(seed)))
