"""Wave-981 convex-geometry-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.grothendieck_const import bench_grothendieck_const
from quant_fund.models.john_ellipsoid import bench_john_ellipsoid
from quant_fund.models.kadison_singer import bench_kadison_singer
from quant_fund.models.loewner_ellipsoid import bench_loewner_ellipsoid
from quant_fund.models.milman_isotropic import bench_milman_isotropic
from quant_fund.models.milman_rev_thm import bench_milman_rev_thm

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


def bench_john_ellipsoid_family(seed: int = _SEED + 36900) -> dict[str, float]:
    return _finite_blob(bench_john_ellipsoid(seed))


def bench_loewner_ellipsoid_family(seed: int = _SEED + 36901) -> dict[str, float]:
    return _finite_blob(bench_loewner_ellipsoid(seed))


def bench_milman_rev_thm_family(seed: int = _SEED + 36902) -> dict[str, float]:
    return _finite_blob(bench_milman_rev_thm(seed))


def bench_grothendieck_const_family(seed: int = _SEED + 36903) -> dict[str, float]:
    return _finite_blob(bench_grothendieck_const(seed))


def bench_kadison_singer_family(seed: int = _SEED + 36904) -> dict[str, float]:
    return _finite_blob(bench_kadison_singer(seed))


def bench_milman_isotropic_family(seed: int = _SEED + 36905) -> dict[str, float]:
    return _finite_blob(bench_milman_isotropic(seed))
