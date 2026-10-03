"""Wave-1007 statistical-mechanics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bose_einstein import bench_bose_einstein
from quant_fund.models.fermi_dirac import bench_fermi_dirac
from quant_fund.models.free_energy import bench_free_energy
from quant_fund.models.gibbs_measure import bench_gibbs_measure
from quant_fund.models.ising_model import bench_ising_model
from quant_fund.models.partition_function import bench_partition_function

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_ising_model_family(seed: int = _SEED + 39500) -> dict[str, float]:
    return _finite_blob(bench_ising_model(seed))


def bench_partition_function_family(seed: int = _SEED + 39501) -> dict[str, float]:
    return _finite_blob(bench_partition_function(seed))


def bench_bose_einstein_family(seed: int = _SEED + 39502) -> dict[str, float]:
    return _finite_blob(bench_bose_einstein(seed))


def bench_fermi_dirac_family(seed: int = _SEED + 39503) -> dict[str, float]:
    return _finite_blob(bench_fermi_dirac(seed))


def bench_gibbs_measure_family(seed: int = _SEED + 39504) -> dict[str, float]:
    return _finite_blob(bench_gibbs_measure(seed))


def bench_free_energy_family(seed: int = _SEED + 39505) -> dict[str, float]:
    return _finite_blob(bench_free_energy(seed))
