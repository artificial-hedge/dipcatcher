"""Wave-514 differential-cohomology bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beilinson_reg import bench_beilinson_reg
from quant_fund.models.cheeger_simons import bench_cheeger_simons
from quant_fund.models.deligne_cohom import bench_deligne_cohom
from quant_fund.models.diff_cohom import bench_diff_cohom
from quant_fund.models.flat_bundle import bench_flat_bundle
from quant_fund.models.secondary_inv import bench_secondary_inv

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


def bench_diff_cohom_family(seed: int = _SEED + 2990) -> dict[str, float]:
    return _floats(_finite_blob("diff_cohom", bench_diff_cohom(seed)))


def bench_cheeger_simons_family(seed: int = _SEED + 2991) -> dict[str, float]:
    return _floats(_finite_blob("cheeger_simons", bench_cheeger_simons(seed)))


def bench_deligne_cohom_family(seed: int = _SEED + 2992) -> dict[str, float]:
    return _floats(_finite_blob("deligne_cohom", bench_deligne_cohom(seed)))


def bench_flat_bundle_family(seed: int = _SEED + 2993) -> dict[str, float]:
    return _floats(_finite_blob("flat_bundle", bench_flat_bundle(seed)))


def bench_beilinson_reg_family(seed: int = _SEED + 2994) -> dict[str, float]:
    return _floats(_finite_blob("beilinson_reg", bench_beilinson_reg(seed)))


def bench_secondary_inv_family(seed: int = _SEED + 2995) -> dict[str, float]:
    return _floats(_finite_blob("secondary_inv", bench_secondary_inv(seed)))
