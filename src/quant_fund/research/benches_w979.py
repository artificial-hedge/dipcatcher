"""Wave-979 ergodic-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.disjointness_dyn import bench_disjointness_dyn
from quant_fund.models.horocycle_flow import bench_horocycle_flow
from quant_fund.models.ratner_thm import bench_ratner_thm
from quant_fund.models.unipotent_ergodic import bench_unipotent_ergodic
from quant_fund.models.van_der_corput import bench_van_der_corput
from quant_fund.models.weyl_equidist import bench_weyl_equidist

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


def bench_weyl_equidist_family(seed: int = _SEED + 36700) -> dict[str, float]:
    return _finite_blob(bench_weyl_equidist(seed))


def bench_van_der_corput_family(seed: int = _SEED + 36701) -> dict[str, float]:
    return _finite_blob(bench_van_der_corput(seed))


def bench_horocycle_flow_family(seed: int = _SEED + 36702) -> dict[str, float]:
    return _finite_blob(bench_horocycle_flow(seed))


def bench_unipotent_ergodic_family(seed: int = _SEED + 36703) -> dict[str, float]:
    return _finite_blob(bench_unipotent_ergodic(seed))


def bench_ratner_thm_family(seed: int = _SEED + 36704) -> dict[str, float]:
    return _finite_blob(bench_ratner_thm(seed))


def bench_disjointness_dyn_family(seed: int = _SEED + 36705) -> dict[str, float]:
    return _finite_blob(bench_disjointness_dyn(seed))
