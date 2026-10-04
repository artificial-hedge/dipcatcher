"""Wave-593 nonabelian-Hodge bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.harmonic_bdl import bench_harmonic_bdl
from quant_fund.models.higgs_bundle2 import bench_higgs_bundle2
from quant_fund.models.hitchin_section import (
    bench_hitchin_section,
)
from quant_fund.models.hodge_moduli import bench_hodge_moduli
from quant_fund.models.nonabelian_hodge import (
    bench_nonabelian_hodge,
)
from quant_fund.models.simpson_corr import bench_simpson_corr

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


def bench_higgs_bundle2_family(
    seed: int = _SEED + 3464,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "higgs_bundle2",
            bench_higgs_bundle2(seed),
        )
    )


def bench_hitchin_section_family(
    seed: int = _SEED + 3465,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hitchin_section",
            bench_hitchin_section(seed),
        )
    )


def bench_simpson_corr_family(
    seed: int = _SEED + 3466,
) -> dict[str, float]:
    return _floats(_finite_blob("simpson_corr", bench_simpson_corr(seed)))


def bench_nonabelian_hodge_family(
    seed: int = _SEED + 3467,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nonabelian_hodge",
            bench_nonabelian_hodge(seed),
        )
    )


def bench_harmonic_bdl_family(
    seed: int = _SEED + 3468,
) -> dict[str, float]:
    return _floats(_finite_blob("harmonic_bdl", bench_harmonic_bdl(seed)))


def bench_hodge_moduli_family(
    seed: int = _SEED + 3469,
) -> dict[str, float]:
    return _floats(_finite_blob("hodge_moduli", bench_hodge_moduli(seed)))
