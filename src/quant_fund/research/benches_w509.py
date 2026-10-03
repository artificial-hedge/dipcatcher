"""Wave-509 Weil-II/l-adic bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.deligne_weil2 import bench_deligne_weil2
from quant_fund.models.etale_site2 import bench_etale_site2
from quant_fund.models.frobenius_action import bench_frobenius_action
from quant_fund.models.groth_lefschetz import bench_groth_lefschetz
from quant_fund.models.l_adic_sheaf import bench_l_adic_sheaf
from quant_fund.models.purity_thm import bench_purity_thm

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


def bench_etale_site2_family(seed: int = _SEED + 2960) -> dict[str, float]:
    return _floats(_finite_blob("etale_site2", bench_etale_site2(seed)))


def bench_l_adic_sheaf_family(seed: int = _SEED + 2961) -> dict[str, float]:
    return _floats(_finite_blob("l_adic_sheaf", bench_l_adic_sheaf(seed)))


def bench_frobenius_action_family(seed: int = _SEED + 2962) -> dict[str, float]:
    return _floats(_finite_blob("frobenius_action", bench_frobenius_action(seed)))


def bench_groth_lefschetz_family(seed: int = _SEED + 2963) -> dict[str, float]:
    return _floats(_finite_blob("groth_lefschetz", bench_groth_lefschetz(seed)))


def bench_deligne_weil2_family(seed: int = _SEED + 2964) -> dict[str, float]:
    return _floats(_finite_blob("deligne_weil2", bench_deligne_weil2(seed)))


def bench_purity_thm_family(seed: int = _SEED + 2965) -> dict[str, float]:
    return _floats(_finite_blob("purity_thm", bench_purity_thm(seed)))
