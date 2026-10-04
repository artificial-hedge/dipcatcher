"""Wave-994 inverse-spectral canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.borg_levinson import bench_borg_levinson
from quant_fund.models.gelfand_levitan import bench_gelfand_levitan
from quant_fund.models.inverse_scattering import bench_inverse_scattering
from quant_fund.models.kdv_isospectral import bench_kdv_isospectral
from quant_fund.models.marchenko_eq import bench_marchenko_eq
from quant_fund.models.trace_formulas import bench_trace_formulas

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


def bench_inverse_scattering_family(seed: int = _SEED + 38200) -> dict[str, float]:
    return _finite_blob(bench_inverse_scattering(seed))


def bench_marchenko_eq_family(seed: int = _SEED + 38201) -> dict[str, float]:
    return _finite_blob(bench_marchenko_eq(seed))


def bench_gelfand_levitan_family(seed: int = _SEED + 38202) -> dict[str, float]:
    return _finite_blob(bench_gelfand_levitan(seed))


def bench_kdv_isospectral_family(seed: int = _SEED + 38203) -> dict[str, float]:
    return _finite_blob(bench_kdv_isospectral(seed))


def bench_trace_formulas_family(seed: int = _SEED + 38204) -> dict[str, float]:
    return _finite_blob(bench_trace_formulas(seed))


def bench_borg_levinson_family(seed: int = _SEED + 38205) -> dict[str, float]:
    return _finite_blob(bench_borg_levinson(seed))
