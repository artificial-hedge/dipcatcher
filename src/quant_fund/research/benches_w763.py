"""Wave-763 mixing/urn bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boneschi_boal import bench_boneschi_boal
from quant_fund.models.bradley_mixing import bench_bradley_mixing
from quant_fund.models.hopf_chain import bench_hopf_chain
from quant_fund.models.ibagimov_mixing import bench_ibagimov_mixing
from quant_fund.models.polya_urn import bench_polya_urn
from quant_fund.models.rosenthal_mom import bench_rosenthal_mom

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


def bench_polya_urn_family(
    seed: int = _SEED + 15200,
) -> dict[str, float]:
    return _floats(_finite_blob("polya_urn", bench_polya_urn(seed)))


def bench_hopf_chain_family(
    seed: int = _SEED + 15201,
) -> dict[str, float]:
    return _floats(_finite_blob("hopf_chain", bench_hopf_chain(seed)))


def bench_boneschi_boal_family(
    seed: int = _SEED + 15202,
) -> dict[str, float]:
    return _floats(_finite_blob("boneschi_boal", bench_boneschi_boal(seed)))


def bench_bradley_mixing_family(
    seed: int = _SEED + 15203,
) -> dict[str, float]:
    return _floats(_finite_blob("bradley_mixing", bench_bradley_mixing(seed)))


def bench_rosenthal_mom_family(
    seed: int = _SEED + 15204,
) -> dict[str, float]:
    return _floats(_finite_blob("rosenthal_mom", bench_rosenthal_mom(seed)))


def bench_ibagimov_mixing_family(
    seed: int = _SEED + 15205,
) -> dict[str, float]:
    return _floats(_finite_blob("ibagimov_mixing", bench_ibagimov_mixing(seed)))
